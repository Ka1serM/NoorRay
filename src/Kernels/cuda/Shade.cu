#include <cuda_fp16.h>

#include "Kernels/Geometry.h"
#include "Kernels/Queues.h"
#include "Kernels/Samplers.h"
#include "Kernels/KernelHelpers.h"

NR_GPU_KERNEL void shadeKernel(const KernelParams params)
{
    const uint32_t index = NR_GPU_LAUNCH_IDX;
    const uint32_t activeCount = params.queues.rayCounts[params.depth];
    const bool inRange = index < activeCount;
    bool continuePath = false;
    PathRayWorkItem continuation{};
    if (inRange)
    {
        const HitWorkItem hit = params.queues.hitQueue[index];
        ShadowWorkItem shadow{};
        shadow.tMin = 1.0f;
        shadow.tMax = 0.0f;
        params.queues.shadowQueue[index] = shadow;
        PathState state = params.queues.pathStates[hit.sampleIndex];
        if (hit.primitiveIndex == InvalidIndex)
        {
            const bool visible = params.scene.environment->visible != 0 &&
                params.scene.renderSettings->transparentBackground == 0;
            state.radiance = state.radiance + state.throughput * environmentRadiance(params.scene, hit.rayDirection, visible);
            if (visible)
                state.packedCounters |= 1u << CounterHitShift;
            params.queues.pathStates[hit.sampleIndex] = state;
        }
        else
        {
            SurfaceData surface = loadSurface(params.scene, hit.instanceIndex, hit.primitiveIndex, hit.baryU, hit.baryV);
            state.packedCounters |= 1u << CounterHitShift;
            if (surface.material != nullptr)
            {
                const GpuMaterial material = *surface.material;
                if (glm::dot(surface.normal, hit.rayDirection) > 0.0f)
                    surface.normal = surface.normal * -1.0f;
                if (glm::dot(surface.geometricNormal, hit.rayDirection) > 0.0f)
                    surface.geometricNormal = surface.geometricNormal * -1.0f;
                const glm::vec3 albedo = sampleTexture(params.scene, material.albedoIndex, surface.uv, material.albedo);
                const glm::vec3 emission = sampleTexture(params.scene, material.emissionIndex, surface.uv, material.emission) * material.emissionStrength;
                const float metallic = fminf(fmaxf(sampleTextureScalar(
                    params.scene, material.metallicIndex, surface.uv, material.metallic, 2), 0.0f), 1.0f);
                const float roughness = fminf(fmaxf(sampleTextureScalar(
                    params.scene, material.roughnessIndex, surface.uv, material.roughness, 1), 0.02f), 1.0f);
                const float transmission = fminf(fmaxf(sampleTextureScalar(
                    params.scene, material.transmissionIndex, surface.uv, material.transmission), 0.0f), 1.0f);
                state.radiance = state.radiance + state.throughput * emission;
                if (state.depth == 0)
                {
                    PrimaryState primary{};
                    primary.primaryAlbedo = albedo;
                    primary.primaryNormal = surface.normal;
                    primary.primaryPosition = surface.position;
                    primary.primaryObjectIndex = surface.objectIndex;
                    params.queues.primaryStates[hit.sampleIndex] = primary;
                }

                uint32_t rng = state.rngState;
                glm::vec3 nextDirection{};
                if (randomFloat(rng) < transmission)
                {
                    nextDirection = hit.rayDirection;
                    state.throughput = state.throughput * material.transmissionColor;
                    state.packedCounters += 1u << CounterTransmissionShift;
                    state.flags = 4u;
                }
                else
                {
                    const float specularProbability = fminf(fmaxf(0.15f + 0.7f * metallic +
                        0.15f * (1.0f - roughness), 0.05f), 0.95f);
                    if (randomFloat(rng) < specularProbability)
                    {
                        nextDirection = glm::normalize(glm::reflect(hit.rayDirection, surface.normal));
                        const glm::vec3 f0 = glm::mix(glm::vec3(0.08f * material.specular), albedo, metallic);
                        state.throughput = state.throughput * (f0 / specularProbability);
                        state.packedCounters += 1u << CounterSpecularShift;
                        state.flags = 2u;
                    }
                    else
                    {
                        nextDirection = cosineHemisphere(surface.normal, rng);
                        state.throughput = state.throughput * (albedo / (1.0f - specularProbability));
                        state.packedCounters += 1u << CounterDiffuseShift;
                        state.flags = 1u;
                    }

                    glm::vec3 lightDirection{};
                    glm::vec3 lightRadiance{};
                    float lightDistance = 1000.0f;
                    const EnvironmentSettings environment = *params.scene.environment;
                    if (environment.directionalIntensity > 0.0f)
                    {
                        lightDirection = glm::normalize(environment.directionalDirection * -1.0f);
                        lightRadiance = glm::vec3(environment.directionalIntensity);
                    }
                    else if (params.scene.lightCount > 0)
                    {
                        const LightGpu light = params.scene.lights[static_cast<uint32_t>(randomFloat(rng) * params.scene.lightCount) % params.scene.lightCount];
                        const glm::vec3 delta = light.position - surface.position;
                        lightDistance = glm::length(delta);
                        lightDirection = delta / lightDistance;
                        const float attenuation = light.useInverseSquaredFalloff != 0
                            ? 1.0f / fmaxf(lightDistance * lightDistance, 0.01f)
                            : 1.0f;
                        lightRadiance = light.color * (light.intensity * attenuation * params.scene.lightCount);
                    }
                    const float cosine = fmaxf(glm::dot(surface.normal, lightDirection), 0.0f);
                    if (cosine > 0.0f && fmaxf(lightRadiance.x, fmaxf(lightRadiance.y, lightRadiance.z)) > 0.0f)
                    {
                        shadow.origin = surface.position + surface.geometricNormal * 0.001f;
                        shadow.direction = lightDirection;
                        shadow.tMin = 0.001f;
                        shadow.tMax = lightDistance - 0.002f;
                        shadow.contribution = state.throughput * albedo * lightRadiance * (cosine / Pi);
                        shadow.sampleIndex = hit.sampleIndex;
                        params.queues.shadowQueue[index] = shadow;
                    }
                }

                state.rngState = rng;
                state.depth++;
                const RenderSettings render = *params.scene.renderSettings;
                const uint32_t diffuse = (state.packedCounters >> CounterDiffuseShift) & 0xffu;
                const uint32_t specular = (state.packedCounters >> CounterSpecularShift) & 0xffu;
                const uint32_t transmitted = (state.packedCounters >> CounterTransmissionShift) & 0xffu;
                continuePath = diffuse <= static_cast<uint32_t>(render.diffuseBounces) &&
                    specular <= static_cast<uint32_t>(render.specularBounces) &&
                    transmitted <= static_cast<uint32_t>(render.transmissionBounces) &&
                    params.depth + 1 < 256;
                if (continuePath && static_cast<int>(state.depth) >= render.russianRouletteStartBounce)
                {
                    const float survival = fminf(fmaxf(fmaxf(state.throughput.x, fmaxf(state.throughput.y, state.throughput.z)), 0.05f), 0.95f);
                    continuePath = randomFloat(state.rngState) <= survival;
                    if (continuePath)
                        state.throughput = state.throughput / survival;
                }
                if (continuePath)
                {
                    continuation.origin = surface.position + surface.geometricNormal *
                        (glm::dot(nextDirection, surface.geometricNormal) >= 0.0f ? 0.001f : -0.001f);
                    continuation.direction = nextDirection;
                    continuation.sampleIndex = hit.sampleIndex;
                }
            }
            params.queues.pathStates[hit.sampleIndex] = state;
        }
    }
    appendRayWarp(params.queues, params.depth + 1, inRange && continuePath, continuation);
}
