#include <cuda_fp16.h>

#include "Raytracing/Geometry.h"
#include "Raytracing/GaussianShading.h"
#include "Raytracing/LightSampling.h"
#include "Raytracing/MisHeuristic.h"
#include "Raytracing/Queues.h"
#include "Raytracing/RgbToSpectrum.h"
#include "Samplers/RandomSampler.h"

NR_GPU inline float analyticLightSelectionWeight(const GpuSceneData& scene)
{
    return scene.analyticLightSelectionWeight;
}

// Samples the BSDF lobe, adds emission, and performs next-event estimation
// against all analytic lights + the environment (with MIS). Shared by mesh
// and Gaussian-splat surface hits so both receive identical direct/indirect
// lighting treatment. Returns the sampled outgoing direction; updates
// state.radiance/throughput/packedCounters/lastBsdfPdfBits and writes a
// shadow-ray work item when the light sample is unoccluded-testable.
NR_GPU inline glm::vec3 shadeBsdfLobe(
    const KernelParams& params,
    const glm::vec3& position,
    const Bsdf& bsdf,
    const SampledSpectrum& emission,
    PathState& state,
    ShadowWorkItem& shadow,
    const uint32_t sampleIndex,
    RandomState& bsdfRng,
    RandomState& lightRng,
    RandomState& shadowRng)
{
    SampledWavelengths& wl = state.wl;
    const SampledSpectrum directThroughput = state.throughput;

    const BsdfSample bsdfSample = bsdf.sample(bsdfRng);
    state.radiance += state.throughput * emission;

    if (bsdfSample.event == BsdfEvent::Transmission)
    {
        state.packedCounters += 1u << CounterTransmissionShift;
        state.etaScale *= bsdfSample.eta * bsdfSample.eta;
    }
    else if (bsdfSample.event == BsdfEvent::Specular)
    {
        state.packedCounters += 1u << CounterSpecularShift;
    }
    else
    {
        state.packedCounters += 1u << CounterDiffuseShift;
    }
    state.throughput *= bsdfSample.weight;
    state.lastBsdfPdfBits = __float_as_uint(
        bsdf.transmission <= 0.0f ? bsdfSample.pdf : 0.0f);

    // Direct light sampling.
    {
        LightSample lightSample{};
        const float analyticWeight = analyticLightSelectionWeight(params.scene);
        const float environmentWeight = bsdf.transmission <= 0.0f
            ? fmaxf(params.scene.environment->importanceWeight, 0.0f) : 0.0f;
        const float totalWeight = analyticWeight + environmentWeight;
        if (totalWeight > 0.0f)
        {
            const float target = randomFloat(lightRng) * totalWeight;
            float selectedWeight = 0.0f;
            bool environmentSelected = false;
            float sampledEnvironmentPdf = 0.0f;
            if (target < analyticWeight && analyticWeight > 0.0f)
            {
                const AnalyticLightSample analytic = sampleAnalyticLight(
                    params.scene, position, lightRng, wl);
                selectedWeight = analytic.selectionPdf * analyticWeight;
                lightSample = analytic.light;
            }
            else if (environmentWeight > 0.0f)
            {
                const EnvironmentSample environmentSample =
                    params.scene.environment->sampleDirection(lightRng);
                if (environmentSample.pdf > 0.0f) {
                    selectedWeight = environmentWeight;
                    environmentSelected = true;
                    sampledEnvironmentPdf = environmentSample.pdf;
                    lightSample.direction = environmentSample.direction;
                    lightSample.distance = 1e16f;
                    lightSample.radiance = params.scene.environment->radiance(
                        params.scene.textures, params.scene.textureCount,
                        environmentSample.direction, false, wl,
                        params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                        params.scene.d65);
                }
            }
            if (selectedWeight > 0.0f) {
                const float selectionPdf = selectedWeight / totalWeight;
                if (environmentSelected) {
                    const float lightPdf = selectionPdf * sampledEnvironmentPdf;
                    const float bsdfPdf = bsdf.pdf(lightSample.direction);
                    lightSample.radiance *= powerHeuristic(lightPdf, bsdfPdf)
                        / fmaxf(lightPdf, 1e-20f);
                } else {
                    lightSample.radiance *= 1.0f / selectionPdf;
                }
            }
        }
        const float cosine = fmaxf(glm::dot(bsdf.shadingNormal, lightSample.direction), 0.0f);
        if (cosine > 0.0f && lightSample.radiance.maxComponent() > 0.0f)
        {
            const SampledSpectrum brdf = bsdf.evaluate(lightSample.direction);
            shadow.origin    = position + bsdf.geometricNormal * 0.001f;
            shadow.direction = lightSample.direction;
            shadow.tMin      = 0.001f;
            shadow.tMax      = lightSample.distance - 0.002f;
            shadow.contribution = directThroughput * brdf * lightSample.radiance * cosine;
            shadow.rngState     = shadowRng;
            shadow.sampleIndex  = sampleIndex;
        }
    }

    if (bsdfSample.event == BsdfEvent::Transmission)
        wl.terminateSecondary();

    return bsdfSample.direction;
}

NR_GPU_KERNEL void shadeKernel(const KernelParams params)
{
    const uint32_t index = NR_GPU_LAUNCH_IDX;
    const uint32_t activeCount = params.queues.rayCounts[params.depth];
    const bool inRange = index < activeCount;
    const bool gaussianDirectColor =
        params.scene.renderSettings.gaussianShadingMode == GaussianShadingMode::DirectColor;
    const bool mayWriteShadowQueue = params.scene.meshInstanceCount > 0 ||
        (!gaussianDirectColor &&
            (params.scene.analyticLightAliasCount > 0
                || params.scene.environment->importanceWeight > 0.0f));
    bool continuePath = false;
    PathRayWorkItem continuation{};
    if (inRange)
    {
        const HitWorkItem hit = params.queues.hitQueue[index];
        ShadowWorkItem shadow{};
        shadow.tMin = 1.0f;
        shadow.tMax = 0.0f;
        if (mayWriteShadowQueue)
            params.queues.shadowQueue[index].tMax = 0.0f;
        PathState state = params.queues.pathStates[hit.sampleIndex];
        SampledWavelengths& wl = state.wl;

        // Gaussian discriminator: instanceIndex >= meshInstanceCount.
        // For gaussian hits instanceIndex encodes meshInstanceCount + gaussianId,
        // A miss also leaves instanceIndex at InvalidIndex, which
        // is >= meshInstanceCount too — exclude it explicitly, or every missed
        // ray reads gaussianShCoeffs[] wildly out of bounds.
        const bool isMiss = hit.instanceIndex == InvalidIndex;
        const bool isGaussianHit = !isMiss && hit.instanceIndex >= params.scene.meshInstanceCount;

        // Gaussian splat: isotropic scattering (no surface normal needed).
        // The Russian roulette any-hit already accepted this splat with
        // probability = density alpha, giving the correct 3DGS over blend.
        // At the scattering point we sample a uniform direction, evaluate
        // next-event estimation against all analytic lights, and continue.
        if (isGaussianHit)
        {
            state.packedCounters |= 1u << CounterHitShift;
            state.packedCounters += 1u << CounterDiffuseShift;

            const uint32_t gaussianId = hit.instanceIndex - params.scene.meshInstanceCount;
            const glm::vec3* shCoeffs = params.scene.gaussianShCoeffs + gaussianId * 16;
            const glm::vec3 viewDirection = glm::normalize(-hit.direction);
            glm::vec3 shRgb = glm::vec3(0.5f) + evaluateGaussianSphericalHarmonics(shCoeffs,
                params.scene.renderSettings.gaussianRenderSphericalHarmonics, viewDirection);
            shRgb.x = fminf(fmaxf(shRgb.x, 0.0f), 1.0f);
            shRgb.y = fminf(fmaxf(shRgb.y, 0.0f), 1.0f);
            shRgb.z = fminf(fmaxf(shRgb.z, 0.0f), 1.0f);
            float shSpectrumC0, shSpectrumC1, shSpectrumC2;
            rgbLookupCoeffs(shRgb, params.scene.spectrumTableScale,
                params.scene.spectrumTableCoeffs, shSpectrumC0, shSpectrumC1, shSpectrumC2);
            const glm::vec3 spectralPolynomial(shSpectrumC0, shSpectrumC1, shSpectrumC2);
            SampledSpectrum albedo;
            for (int i = 0; i < NrSpectrumSamples; ++i)
                albedo.values[i] = rgbSigmoidEval(spectralPolynomial.x,
                    spectralPolynomial.y, spectralPolynomial.z, wl.lambda[i]);

            if (params.scene.renderSettings.gaussianShadingMode == GaussianShadingMode::DirectColor)
            {
                state.radiance += state.throughput * albedo;
                params.queues.pathStates[hit.sampleIndex] = state;
                continuePath = false;
            }
            else
            {
            const float inv4Pi = 0.07957747154594767f; // 1/(4*pi)
            const glm::vec3 gaussianPos = hit.gaussianData;

            const RandomState bounceKey = state.rngState;
            state.rngState = advanceRandomSequence(bounceKey);
            RandomState bsdfRng = forkRandom(bounceKey, RandomStream::Bsdf);
            RandomState lightRng = forkRandom(bounceKey, RandomStream::Light);
            RandomState shadowRng = forkRandom(bounceKey, RandomStream::Shadow);
            RandomState rouletteRng = forkRandom(bounceKey, RandomStream::Roulette);

            // ── Scatter direction (uniform sphere, isotropic) ──────
            const float theta = 2.0f * EnvironmentPi * randomFloat(bsdfRng);
            const float z = 2.0f * randomFloat(bsdfRng) - 1.0f;
            const float radius = sqrtf(fmaxf(1.0f - z * z, 0.0f));
            float sinTheta = 0.0f;
            float cosTheta = 1.0f;
            sincosf(theta, &sinTheta, &cosTheta);
            const glm::vec3 nextDirection(radius * cosTheta, z, radius * sinTheta);

            // ── NEE: sample analytic lights ────────────────────────
            {
                LightSample lightSample{};
                const float analyticWeight = analyticLightSelectionWeight(params.scene);
                const float environmentWeight = fmaxf(
                    params.scene.environment->importanceWeight, 0.0f);
                const float totalWeight = analyticWeight + environmentWeight;
                if (totalWeight > 0.0f)
                {
                    const float target = randomFloat(lightRng) * totalWeight;
                    float selectedWeight = 0.0f;
                    bool environmentSelected = false;
                    float sampledEnvironmentPdf = 0.0f;
                    if (target < analyticWeight && analyticWeight > 0.0f)
                    {
                        const AnalyticLightSample analytic = sampleAnalyticLight(
                            params.scene, gaussianPos, lightRng, wl);
                        selectedWeight = analytic.selectionPdf * analyticWeight;
                        lightSample = analytic.light;
                    }
                    else if (environmentWeight > 0.0f)
                    {
                        const EnvironmentSample environmentSample =
                            params.scene.environment->sampleDirection(lightRng);
                        if (environmentSample.pdf > 0.0f)
                        {
                            selectedWeight = environmentWeight;
                            environmentSelected = true;
                            sampledEnvironmentPdf = environmentSample.pdf;
                            lightSample.direction = environmentSample.direction;
                            lightSample.distance = 1e16f;
                            lightSample.radiance = params.scene.environment->radiance(
                                params.scene.textures, params.scene.textureCount,
                                environmentSample.direction, false, wl,
                                params.scene.spectrumTableScale,
                                params.scene.spectrumTableCoeffs, params.scene.d65);
                        }
                    }
                    if (selectedWeight > 0.0f && lightSample.radiance.maxComponent() > 0.0f)
                    {
                        const float selectionPdf = selectedWeight / totalWeight;
                        const SampledSpectrum brdf = albedo * inv4Pi;
                        if (environmentSelected)
                        {
                            const float lightPdf = selectionPdf * sampledEnvironmentPdf;
                            lightSample.radiance *= powerHeuristic(lightPdf, inv4Pi)
                                / fmaxf(lightPdf, 1e-20f);
                        }
                        else
                        {
                            lightSample.radiance *= 1.0f / selectionPdf;
                        }
                        shadow.origin    = gaussianPos + lightSample.direction * 0.001f;
                        shadow.direction = lightSample.direction;
                        shadow.tMin      = 0.001f;
                        shadow.tMax      = lightSample.distance - 0.002f;
                        shadow.contribution =
                            state.throughput * brdf * lightSample.radiance;
                        shadow.rngState     = shadowRng;
                        shadow.sampleIndex  = hit.sampleIndex;
                        params.queues.shadowQueue[index] = shadow;
                    }
                }
            }

            state.throughput *= albedo;
            state.lastBsdfPdfBits = __float_as_uint(inv4Pi);

            state.depth++;
            continuePath = true;
            if (static_cast<int>(state.depth) >= params.scene.renderSettings.russianRouletteStartBounce)
            {
                const float survival = fminf(fmaxf(
                    state.throughput.maxComponent(), 0.05f), 0.95f);
                continuePath = randomFloat(rouletteRng) <= survival;
                if (continuePath)
                    state.throughput *= (1.0f / survival);
            }
            if (continuePath)
            {
                continuation.origin = gaussianPos + nextDirection * 0.001f;
                continuation.direction = nextDirection;
                continuation.sampleIndex = hit.sampleIndex;
            }
            params.queues.pathStates[hit.sampleIndex] = state;
            }
        }
        else if (isMiss)
        {
            const bool cameraRay       = state.depth == 0;
            const bool backgroundVisible = params.scene.environment->visible != 0 &&
                !params.scene.renderSettings.transparentBackground;
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
                            * params.scene.environment->pdf(hit.direction);
                        misWeight = powerHeuristic(bsdfPdf, lightPdf);
                    }
                }
                state.radiance += state.throughput *
                    params.scene.environment->radiance(
                        params.scene.textures, params.scene.textureCount,
                        hit.direction, cameraRay, wl,
                        params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                        params.scene.d65)
                    * misWeight;
            }
            if (cameraRay && backgroundVisible)
                state.packedCounters |= 1u << CounterHitShift;
            params.queues.pathStates[hit.sampleIndex] = state;
        }
        else
        {
            SurfaceData surface = loadSurface(params.scene, hit.instanceIndex, hit.primitiveIndex, hit.attribute0, hit.attribute1);
            const Material& material = *surface.material;
            const glm::vec3 originalGeometricNormal = surface.geometricNormal;
            glm::vec3 shadingNormal = applyNormalMap(
                material, params.scene.textures, surface.uv, surface.tangent, surface.normal);
            const RandomState bounceKey = state.rngState;
            state.rngState = advanceRandomSequence(bounceKey);
            RandomState opacityRng = forkRandom(bounceKey, RandomStream::Opacity);
            RandomState bsdfRng = forkRandom(bounceKey, RandomStream::Bsdf);
            RandomState lightRng = forkRandom(bounceKey, RandomStream::Light);
            RandomState shadowRng = forkRandom(bounceKey, RandomStream::Shadow);
            RandomState rouletteRng = forkRandom(bounceKey, RandomStream::Roulette);
            glm::vec3 nextDirection{};
            float opacity = material.opacity;
            if (material.opacityIndex >= 0)
                opacity *= params.scene.textures[material.opacityIndex].sample(surface.uv).w;
            opacity = fminf(fmaxf(opacity, 0.0f), 1.0f);
            const bool transparentSurface = randomFloat(opacityRng) > opacity;
            if (transparentSurface)
            {
                nextDirection = hit.direction;
            }
            else
            {
                state.packedCounters |= 1u << CounterHitShift;
                if (glm::dot(shadingNormal, hit.direction) > 0.0f)
                    shadingNormal = -shadingNormal;
                if (glm::dot(surface.geometricNormal, hit.direction) > 0.0f)
                    surface.geometricNormal = -surface.geometricNormal;
                const glm::vec3 viewDirection = -hit.direction;

                shadingNormal = Bsdf::clampShadingNormal(
                    surface.geometricNormal, shadingNormal, viewDirection);

                const Bsdf bsdf = material.makeBsdf(
                    params.scene.textures, surface.uv,
                    viewDirection, originalGeometricNormal, shadingNormal, wl,
                    params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                    params.scene.openPbrLuts);
                const SampledSpectrum emission = material.emissionSpectral(
                    params.scene.textures, surface.uv, wl,
                    params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                    params.scene.d65);
                nextDirection = shadeBsdfLobe(
                    params, surface.position, bsdf, emission, state, shadow,
                    hit.sampleIndex, bsdfRng, lightRng, shadowRng);
                if (shadow.tMax > shadow.tMin)
                    params.queues.shadowQueue[index] = shadow;
            }

            state.depth++;
            continuePath = true;
            if (static_cast<int>(state.depth) >= params.scene.renderSettings.russianRouletteStartBounce)
            {
                const float survival = fminf(fmaxf(
                    state.throughput.maxComponent() * state.etaScale, 0.05f), 0.95f);
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
            params.queues.pathStates[hit.sampleIndex] = state;
        }
    }
    appendRayWarp(params.queues, params.depth + 1, inRange && continuePath, continuation);
}
