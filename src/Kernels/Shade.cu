#include <cuda_fp16.h>

#include "OpenPbr/OpenPbrSurface.h"
#include "Raytracing/Geometry.h"
#include "Raytracing/KernelHelpers.h"
#include "Raytracing/Queues.h"
#include "Samplers/RandomSampler.h"

NR_GPU inline float analyticLightSelectionWeight(const GpuSceneData& scene)
{
    float weight = 0.0f;
    for (uint32_t i = 0; i < scene.pointLightCount; ++i)
        weight += scene.pointLights[i].selectionWeight();
    for (uint32_t i = 0; i < scene.spotLightCount; ++i)
        weight += scene.spotLights[i].selectionWeight();
    for (uint32_t i = 0; i < scene.rectLightCount; ++i)
        weight += scene.rectLights[i].selectionWeight();
    for (uint32_t i = 0; i < scene.directionalLightCount; ++i)
        weight += scene.directionalLights[i].selectionWeight();
    return weight;
}

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

        // Reconstruct sampled wavelengths from PathState.
        SampledWavelengths wl;
        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            wl.lambda[i] = state.lambda[i];
            wl.pdf[i]    = state.lambdaPdf[i];
        }

        if (hit.primitiveIndex == InvalidIndex)
        {
            const bool cameraRay       = state.depth == 0;
            const bool backgroundVisible = params.scene.environment->visible != 0 &&
                params.scene.renderSettings.transparentBackground == 0;
            if (!cameraRay || backgroundVisible) {
                float misWeight = 1.0f;
                if (!cameraRay) {
                    const float bsdfPdf = __uint_as_float(state.lastBsdfPdfBits);
                    const float environmentWeight = fmaxf(
                        params.scene.environment->importanceWeight, 0.0f);
                    const float totalWeight = analyticLightSelectionWeight(params.scene)
                        + environmentWeight;
                    if (bsdfPdf > 0.0f && environmentWeight > 0.0f && totalWeight > 0.0f) {
                        const float lightPdf = (environmentWeight / totalWeight)
                            * environmentPdf(*params.scene.environment, hit.rayDirection);
                        misWeight = powerHeuristic(bsdfPdf, lightPdf);
                    }
                }
                state.radiance += state.throughput *
                    environmentRadiance(params.scene, hit.rayDirection, cameraRay, wl)
                    * misWeight;
            }
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
                glm::vec3 shadingNormal = applyNormalMap(
                    material, params.scene.textures, surface.uv, surface.tangent, surface.normal);
                // Independent per-bounce streams keep conditional BSDF/light
                // branches from shifting opacity, lighting, or RR dimensions.
                const RandomState bounceKey = state.rngState;
                state.rngState = splitMix64(bounceKey);
                RandomState opacityRng = seedRandom(bounceKey ^ 0x243f6a8885a308d3ull);
                RandomState bsdfRng = seedRandom(bounceKey ^ 0x13198a2e03707344ull);
                RandomState lightRng = seedRandom(bounceKey ^ 0xa4093822299f31d0ull);
                RandomState shadowRng = seedRandom(bounceKey ^ 0x082efa98ec4e6c89ull);
                RandomState rouletteRng = seedRandom(bounceKey ^ 0x452821e638d01377ull);
                const SampledSpectrum directThroughput = state.throughput;
                glm::vec3 nextDirection{};
                bool validContinuation = true;
                float opacity = material.opacity;
                if (material.opacityIndex >= 0)
                    opacity *= params.scene.textures[material.opacityIndex].sample(surface.uv).w;
                opacity = fminf(fmaxf(opacity, 0.0f), 1.0f);
                const bool transparentSurface = randomFloat(opacityRng) > opacity;
                if (transparentSurface)
                {
                    nextDirection = hit.rayDirection;
                    state.flags = 0u;
                }
                else
                {
                    state.packedCounters |= 1u << CounterHitShift;
                    // OpenPBR handles back-facing internally via the basis normal;
                    // do NOT face-forward the shading normal.  Keep geometricNormal
                    // face-forwarded for continuation/shadow ray offsetting.
                    const glm::vec3 nonFfShadingNormal = shadingNormal;
                    if (glm::dot(surface.geometricNormal, hit.rayDirection) > 0.0f)
                        surface.geometricNormal = -surface.geometricNormal;
                    const glm::vec3 viewDirection = -hit.rayDirection;

                    const BsdfSample bsdfSample = nr::openpbr::sample(
                        material, params.scene.textures, surface.uv,
                        viewDirection, nonFfShadingNormal,
                        surface.tangent, bsdfRng, wl,
                        1.0f, state.throughput);

                    state.radiance += state.throughput * bsdfSample.emission;

                    validContinuation = bsdfSample.pdf > 0.0f &&
                        glm::dot(bsdfSample.direction, bsdfSample.direction) > 0.0f;

                    if (validContinuation && bsdfSample.event == BsdfEvent::Transmission)
                    {
                        state.packedCounters += 1u << CounterTransmissionShift;
                        state.flags = 4u;
                        state.etaScale *= bsdfSample.eta * bsdfSample.eta;
                    }
                    else if (validContinuation && bsdfSample.event == BsdfEvent::Specular)
                    {
                        state.packedCounters += 1u << CounterSpecularShift;
                        state.flags = 2u;
                    }
                    else if (validContinuation)
                    {
                        state.packedCounters += 1u << CounterDiffuseShift;
                        state.flags = 1u;
                    }
                    if (validContinuation)
                    {
                        nextDirection = bsdfSample.direction;
                        state.throughput *= bsdfSample.weight;
                    }
                    // Environment NEE is deliberately disabled for transmissive
                    // materials below. With no competing light-sampling strategy,
                    // weighting an eventual environment hit against its PDF would
                    // discard energy.
                    state.lastBsdfPdfBits = __float_as_uint(
                        validContinuation && material.transmission <= 0.0f
                            ? bsdfSample.pdf : 0.0f);

                    // Direct light sampling.
                    {
                        LightSample lightSample{};
                        const uint32_t pl    = params.scene.pointLightCount;
                        const uint32_t sl    = params.scene.spotLightCount;
                        const uint32_t rl    = params.scene.rectLightCount;
                        const uint32_t dl    = params.scene.directionalLightCount;
                        const float analyticWeight = analyticLightSelectionWeight(params.scene);
                        const float environmentWeight = material.transmission <= 0.0f
                            ? fmaxf(params.scene.environment->importanceWeight, 0.0f) : 0.0f;
                        const float totalWeight = analyticWeight + environmentWeight;
                        if (totalWeight > 0.0f)
                        {
                            float target = randomFloat(lightRng) * totalWeight;
                            float selectedWeight = 0.0f;
                            bool environmentSelected = false;
                            float sampledEnvironmentPdf = 0.0f;
                            for (uint32_t i = 0; i < pl && selectedWeight == 0.0f; ++i) {
                                const float w = params.scene.pointLights[i].selectionWeight();
                                if (target < w) {
                                    selectedWeight = w;
                                    lightSample = params.scene.pointLights[i].sampleLi(
                                        surface.position, lightRng, wl,
                                        params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                                        params.scene.d65);
                                } else target -= w;
                            }
                            for (uint32_t i = 0; i < sl && selectedWeight == 0.0f; ++i) {
                                const float w = params.scene.spotLights[i].selectionWeight();
                                if (target < w) {
                                    selectedWeight = w;
                                    lightSample = params.scene.spotLights[i].sampleLi(
                                        surface.position, lightRng, wl,
                                        params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                                        params.scene.d65);
                                } else target -= w;
                            }
                            for (uint32_t i = 0; i < rl && selectedWeight == 0.0f; ++i) {
                                const float w = params.scene.rectLights[i].selectionWeight();
                                if (target < w) {
                                    selectedWeight = w;
                                    lightSample = params.scene.rectLights[i].sampleLi(
                                        surface.position, lightRng, wl,
                                        params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                                        params.scene.d65);
                                } else target -= w;
                            }
                            for (uint32_t i = 0; i < dl && selectedWeight == 0.0f; ++i) {
                                const float w = params.scene.directionalLights[i].selectionWeight();
                                if (target < w) {
                                    selectedWeight = w;
                                    lightSample = params.scene.directionalLights[i].sampleLi(
                                        surface.position, lightRng, wl,
                                        params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                                        params.scene.d65);
                                } else target -= w;
                            }
                            if (selectedWeight == 0.0f && environmentWeight > 0.0f) {
                                const EnvironmentDirectionSample environmentSample =
                                    sampleEnvironmentDirection(*params.scene.environment, lightRng);
                                if (environmentSample.pdf > 0.0f) {
                                    selectedWeight = environmentWeight;
                                    environmentSelected = true;
                                    sampledEnvironmentPdf = environmentSample.pdf;
                                    lightSample.direction = environmentSample.direction;
                                    lightSample.distance = 1e16f;
                                    lightSample.radiance = environmentRadiance(
                                        params.scene, environmentSample.direction, false, wl);
                                }
                            }
                            if (selectedWeight > 0.0f) {
                                const float selectionPdf = selectedWeight / totalWeight;
                                if (environmentSelected) {
                                    const float lightPdf = selectionPdf * sampledEnvironmentPdf;
                                    const float bsdfPdf = nr::openpbr::pdf(
                                        material, params.scene.textures, surface.uv,
                                        viewDirection, lightSample.direction,
                                        nonFfShadingNormal, surface.tangent, state.throughput);
                                    lightSample.radiance *= powerHeuristic(lightPdf, bsdfPdf)
                                        / fmaxf(lightPdf, 1e-20f);
                                } else {
                                    lightSample.radiance *= 1.0f / selectionPdf;
                                }
                            }
                        }
                        if (lightSample.radiance.maxComponent() > 0.0f)
                        {
                            const SampledSpectrum bsdfCosine = nr::openpbr::evaluate(
                                material, params.scene.textures, surface.uv,
                                viewDirection, lightSample.direction,
                                nonFfShadingNormal, surface.tangent, wl,
                                1.0f, state.throughput);
                            if (bsdfCosine.maxComponent() > 0.0f)
                            {
                                const float side = glm::dot(
                                    lightSample.direction, surface.geometricNormal) >= 0.0f
                                    ? 1.0f : -1.0f;
                                shadow.origin = surface.position
                                    + surface.geometricNormal * (0.001f * side);
                                shadow.direction = lightSample.direction;
                                shadow.tMin = 0.001f;
                                shadow.tMax = lightSample.distance - 0.002f;
                                shadow.contribution = directThroughput
                                    * bsdfCosine * lightSample.radiance;
                                shadow.rngState = shadowRng;
                                shadow.sampleIndex = hit.sampleIndex;
                                params.queues.shadowQueue[index] = shadow;
                            }
                        }
                    }


                }

                state.depth++;
                const RenderSettings render = params.scene.renderSettings;
                const uint32_t diffuse    = (state.packedCounters >> CounterDiffuseShift)     & 0xffu;
                const uint32_t specular   = (state.packedCounters >> CounterSpecularShift)    & 0xffu;
                const uint32_t transmitted = (state.packedCounters >> CounterTransmissionShift) & 0xffu;
                continuePath = validContinuation &&
                               diffuse   <= static_cast<uint32_t>(render.diffuseBounces)    &&
                               specular  <= static_cast<uint32_t>(render.specularBounces)   &&
                               transmitted <= static_cast<uint32_t>(render.transmissionBounces) &&
                               params.depth + 1 < 256;
                if (continuePath && static_cast<int>(state.depth) >= render.russianRouletteStartBounce)
                {
                    const float survival = fminf(fmaxf(
                        spectrumY(state.throughput, wl, params.scene.cieY)
                            * state.etaScale, 0.05f), 0.95f);
                    continuePath = randomFloat(rouletteRng) <= survival;
                    if (continuePath)
                        state.throughput *= (1.0f / survival);
                }
                if (continuePath)
                {
                    continuation.origin = surface.position + surface.geometricNormal *
                        (glm::dot(nextDirection, surface.geometricNormal) >= 0.0f ? 0.001f : -0.001f);
                    continuation.direction  = nextDirection;
                    continuation.sampleIndex = hit.sampleIndex;
                }
            }
            params.queues.pathStates[hit.sampleIndex] = state;
        }
    }
    appendRayWarp(params.queues, params.depth + 1, inRange && continuePath, continuation);
}
