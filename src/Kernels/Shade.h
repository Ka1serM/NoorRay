#pragma once

#include <cuda_runtime_api.h>

#include "Kernels/SceneData.h"

void launchShade(const KernelParams& params, cudaStream_t stream);

#ifdef __CUDACC__

#include <cuda_fp16.h>

#include "Kernels/Geometry.h"
#include "Kernels/Math.h"
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
            const bool visible = params.scene.settings->environment.visible != 0 &&
                params.scene.settings->renderSettings.transparentBackground == 0;
            state.radiance = add(state.radiance,
                multiply(state.throughput, environmentRadiance(params.scene, hit.rayDirection, visible)));
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
                if (dot3(surface.normal, hit.rayDirection) > 0.0f)
                    surface.normal = multiply(surface.normal, -1.0f);
                if (dot3(surface.geometricNormal, hit.rayDirection) > 0.0f)
                    surface.geometricNormal = multiply(surface.geometricNormal, -1.0f);
                const glm::vec3 albedo = sampleTexture(params.scene, material.albedoIndex, surface.uv, material.albedo);
                const glm::vec3 emission = multiply(
                    sampleTexture(params.scene, material.emissionIndex, surface.uv, material.emission),
                    material.emissionStrength);
                const float metallic = fminf(fmaxf(sampleTextureScalar(
                    params.scene, material.metallicIndex, surface.uv, material.metallic, 2), 0.0f), 1.0f);
                const float roughness = fminf(fmaxf(sampleTextureScalar(
                    params.scene, material.roughnessIndex, surface.uv, material.roughness, 1), 0.02f), 1.0f);
                const float transmission = fminf(fmaxf(sampleTextureScalar(
                    params.scene, material.transmissionIndex, surface.uv, material.transmission), 0.0f), 1.0f);
                state.radiance = add(state.radiance, multiply(state.throughput, emission));
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
                    state.throughput = multiply(state.throughput, material.transmissionColor);
                    state.packedCounters += 1u << CounterTransmissionShift;
                    state.flags = 4u;
                }
                else
                {
                    const float specularProbability = fminf(fmaxf(0.15f + 0.7f * metallic +
                        0.15f * (1.0f - roughness), 0.05f), 0.95f);
                    if (randomFloat(rng) < specularProbability)
                    {
                        nextDirection = normalize3(reflect3(hit.rayDirection, surface.normal));
                        const glm::vec3 f0 = lerp3(makeVec3(0.08f * material.specular), albedo, metallic);
                        state.throughput = multiply(state.throughput, divide(f0, specularProbability));
                        state.packedCounters += 1u << CounterSpecularShift;
                        state.flags = 2u;
                    }
                    else
                    {
                        nextDirection = cosineHemisphere(surface.normal, rng);
                        state.throughput = multiply(state.throughput, divide(albedo, 1.0f - specularProbability));
                        state.packedCounters += 1u << CounterDiffuseShift;
                        state.flags = 1u;
                    }

                    glm::vec3 lightDirection{};
                    glm::vec3 lightRadiance{};
                    float lightDistance = 1000.0f;
                    const EnvironmentSettings environment = params.scene.settings->environment;
                    if (environment.directionalIntensity > 0.0f)
                    {
                        lightDirection = normalize3(multiply(environment.directionalDirection, -1.0f));
                        lightRadiance = makeVec3(environment.directionalIntensity);
                    }
                    else if (params.scene.lightCount > 0)
                    {
                        const LightGpu light = params.scene.lights[static_cast<uint32_t>(randomFloat(rng) * params.scene.lightCount) % params.scene.lightCount];
                        const glm::vec3 delta = subtract(light.position, surface.position);
                        lightDistance = length3(delta);
                        lightDirection = divide(delta, lightDistance);
                        const float attenuation = light.useInverseSquaredFalloff != 0
                            ? 1.0f / fmaxf(lightDistance * lightDistance, 0.01f)
                            : 1.0f;
                        lightRadiance = multiply(light.color, light.intensity * attenuation * params.scene.lightCount);
                    }
                    const float cosine = fmaxf(dot3(surface.normal, lightDirection), 0.0f);
                    if (cosine > 0.0f && maxComponent(lightRadiance) > 0.0f)
                    {
                        shadow.origin = add(surface.position, multiply(surface.geometricNormal, 0.001f));
                        shadow.direction = lightDirection;
                        shadow.tMin = 0.001f;
                        shadow.tMax = lightDistance - 0.002f;
                        shadow.contribution = multiply(multiply(state.throughput, albedo),
                            multiply(lightRadiance, cosine / Pi));
                        shadow.sampleIndex = hit.sampleIndex;
                        params.queues.shadowQueue[index] = shadow;
                    }
                }

                state.rngState = rng;
                state.depth++;
                const RenderSettings render = params.scene.settings->renderSettings;
                const uint32_t diffuse = (state.packedCounters >> CounterDiffuseShift) & 0xffu;
                const uint32_t specular = (state.packedCounters >> CounterSpecularShift) & 0xffu;
                const uint32_t transmitted = (state.packedCounters >> CounterTransmissionShift) & 0xffu;
                continuePath = diffuse <= static_cast<uint32_t>(render.diffuseBounces) &&
                    specular <= static_cast<uint32_t>(render.specularBounces) &&
                    transmitted <= static_cast<uint32_t>(render.transmissionBounces) &&
                    params.depth + 1 < 256;
                if (continuePath && static_cast<int>(state.depth) >= render.russianRouletteStartBounce)
                {
                    const float survival = fminf(fmaxf(maxComponent(state.throughput), 0.05f), 0.95f);
                    continuePath = randomFloat(state.rngState) <= survival;
                    if (continuePath)
                        state.throughput = divide(state.throughput, survival);
                }
                if (continuePath)
                {
                    continuation.origin = add(surface.position, multiply(surface.geometricNormal,
                        dot3(nextDirection, surface.geometricNormal) >= 0.0f ? 0.001f : -0.001f));
                    continuation.direction = nextDirection;
                    continuation.sampleIndex = hit.sampleIndex;
                }
            }
            params.queues.pathStates[hit.sampleIndex] = state;
        }
    }
    appendRayWarp(params.queues, params.depth + 1, inRange && continuePath, continuation);
}

void launchShade(const KernelParams& params, const cudaStream_t stream)
{
    constexpr uint32_t blockSize = 256;
    shadeKernel<<<(params.queues.capacity + blockSize - 1) / blockSize, blockSize, 0, stream>>>(params);
}

#endif
