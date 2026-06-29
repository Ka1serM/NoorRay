#include <cuda_fp16.h>

#include "Kernels/Geometry.h"
#include "Kernels/Bsdf.h"
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
            const bool cameraRay = state.depth == 0;
            const bool backgroundVisible = params.scene.environment->visible != 0 &&
                params.scene.renderSettings->transparentBackground == 0;
            if (!cameraRay || backgroundVisible)
                state.radiance = state.radiance + state.throughput *
                    environmentRadiance(params.scene, hit.rayDirection, cameraRay);
            if (cameraRay && backgroundVisible)
                state.packedCounters |= 1u << CounterHitShift;
            params.queues.pathStates[hit.sampleIndex] = state;
        }
        else
        {
            SurfaceData surface = loadSurface(params.scene, hit.instanceIndex, hit.primitiveIndex, hit.baryU, hit.baryV);
            if (surface.material != nullptr)
            {
                const Material material = *surface.material;
                const glm::vec3 originalGeometricNormal = surface.geometricNormal;
                glm::vec3 shadingNormal = surface.normal;
                if (material.normalIndex >= 0)
                {
                    const float4 encoded = tex2D<float4>(
                        params.scene.textures[material.normalIndex], surface.uv.x, surface.uv.y);
                    const glm::vec3 tangentNormal = glm::vec3(encoded.x, encoded.y, encoded.z) * 2.0f - 1.0f;
                    const glm::vec3 tangent = glm::normalize(surface.tangent);
                    const glm::vec3 bitangent = glm::normalize(glm::cross(shadingNormal, tangent));
                    shadingNormal = glm::normalize(
                        tangent * tangentNormal.x + bitangent * tangentNormal.y
                        + shadingNormal * tangentNormal.z);
                }
                const glm::vec3 albedo = sampleTexture(params.scene, material.albedoIndex, surface.uv, material.albedo);
                const glm::vec3 emission = sampleTexture(params.scene, material.emissionIndex, surface.uv, material.emission) * material.emissionStrength;
                const float metallic = fminf(fmaxf(sampleTextureScalar(
                    params.scene, material.metallicIndex, surface.uv, material.metallic, 2), 0.0f), 1.0f);
                const float roughness = fminf(fmaxf(sampleTextureScalar(
                    params.scene, material.roughnessIndex, surface.uv, material.roughness, 1), 0.045f), 1.0f);
                const float transmission = fminf(fmaxf(sampleTextureScalar(
                    params.scene, material.transmissionIndex, surface.uv, material.transmission), 0.0f), 1.0f);
                const float specularFactor = fminf(fmaxf(sampleTextureScalar(
                    params.scene, material.specularIndex, surface.uv, material.specular), 0.0f), 1.0f);
                uint32_t rng = state.rngState;
                const glm::vec3 directThroughput = state.throughput;
                glm::vec3 nextDirection{};
                float opacity = material.opacity;
                if (material.opacityIndex >= 0)
                {
                    const float4 opacitySample = tex2D<float4>(
                        params.scene.textures[material.opacityIndex], surface.uv.x, surface.uv.y);
                    opacity *= opacitySample.w;
                }
                opacity = fminf(fmaxf(opacity, 0.0f), 1.0f);
                const bool transparentSurface = randomFloat(rng) > opacity;
                if (transparentSurface)
                {
                    nextDirection = hit.rayDirection;
                    state.flags = 0u;
                }
                else
                {
                    state.packedCounters |= 1u << CounterHitShift;
                    if (glm::dot(shadingNormal, hit.rayDirection) > 0.0f)
                        shadingNormal = -shadingNormal;
                    if (glm::dot(surface.geometricNormal, hit.rayDirection) > 0.0f)
                        surface.geometricNormal = -surface.geometricNormal;
                    const glm::vec3 viewDirection = -hit.rayDirection;

                    state.radiance = state.radiance + state.throughput * emission;
                    if (state.depth == 0)
                    {
                        PrimaryState primary{};
                        primary.primaryAlbedo = albedo;
                        primary.primaryNormal = shadingNormal;
                        primary.primaryPosition = surface.position;
                        primary.primaryObjectIndex = surface.objectIndex;
                        params.queues.primaryStates[hit.sampleIndex] = primary;
                    }

                    const bool dielectric = randomFloat(rng) < transmission;
                    BsdfSample bsdfSample{};
                    if (dielectric)
                    {
                        bsdfSample = sampleDielectricBsdf(
                            viewDirection, originalGeometricNormal, shadingNormal,
                            roughness, material.ior, material.transmissionColor, rng);
                        state.packedCounters += 1u << CounterTransmissionShift;
                        state.flags = 4u;
                    }
                    else
                    {
                        bsdfSample = sampleOpaqueBsdf(
                            viewDirection, shadingNormal, albedo, metallic,
                            specularFactor, roughness, rng);
                        if (bsdfSample.event == BsdfEvent::Specular)
                        {
                            state.packedCounters += 1u << CounterSpecularShift;
                            state.flags = 2u;
                        }
                        else
                        {
                            state.packedCounters += 1u << CounterDiffuseShift;
                            state.flags = 1u;
                        }
                    }
                    nextDirection = bsdfSample.direction;
                    state.throughput = state.throughput * bsdfSample.weight;

                    {
                        LightSample lightSample{};
                        const uint32_t pl = params.scene.pointLightCount;
                        const uint32_t sl = params.scene.spotLightCount;
                        const uint32_t rl = params.scene.rectLightCount;
                        const uint32_t dl = params.scene.directionalLightCount;
                        const uint32_t total = pl + sl + rl + dl;
                        if (total > 0)
                        {
                            uint32_t choice = static_cast<uint32_t>(
                                randomFloat(rng) * static_cast<float>(total));
                            if (choice >= total)
                                choice = total - 1;
                            if (choice < pl)
                                lightSample = params.scene.pointLights[choice].sampleLi(surface.position, rng);
                            else if (choice < pl + sl)
                                lightSample = params.scene.spotLights[choice - pl].sampleLi(surface.position, rng);
                            else if (choice < pl + sl + rl)
                                lightSample = params.scene.rectLights[choice - pl - sl].sampleLi(surface.position, rng);
                            else
                                lightSample = params.scene.directionalLights[choice - pl - sl - rl]
                                    .sampleLi(surface.position, rng);
                            lightSample.radiance *= static_cast<float>(total);
                        }
                        const float cosine = fmaxf(glm::dot(shadingNormal, lightSample.direction), 0.0f);
                        if (cosine > 0.0f && fmaxf(lightSample.radiance.x,
                            fmaxf(lightSample.radiance.y, lightSample.radiance.z)) > 0.0f)
                        {
                            const glm::vec3 brdf = evaluateDirectBsdf(
                                shadingNormal, viewDirection, lightSample.direction,
                                albedo, metallic, specularFactor, roughness,
                                transmission, material.ior);
                            shadow.origin = surface.position + surface.geometricNormal * 0.001f;
                            shadow.direction = lightSample.direction;
                            shadow.tMin = 0.001f;
                            shadow.tMax = lightSample.distance - 0.002f;
                            shadow.contribution = directThroughput * brdf
                                * lightSample.radiance * cosine;
                            shadow.sampleIndex = hit.sampleIndex;
                            params.queues.shadowQueue[index] = shadow;
                        }
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
