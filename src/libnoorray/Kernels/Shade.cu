#include <cuda_fp16.h>

#include <glm/gtc/quaternion.hpp>

#include "Raytracing/Geometry.h"
#include "Raytracing/MisHeuristic.h"
#include "Raytracing/Queues.h"
#include "Raytracing/RgbToSpectrum.h"
#include "Samplers/RandomSampler.h"

NR_GPU inline float analyticLightSelectionWeight(const GpuSceneData& scene)
{
    return scene.analyticLightSelectionWeight;
}

NR_GPU inline glm::vec3 evaluateGaussianSphericalHarmonics(
    const glm::vec3* coefficients, const SphericalHarmonicsOrder order, const glm::vec3 direction)
{
    constexpr float C0 = 0.28209479177387814f;
    constexpr float C1 = 0.4886025119029199f;
    constexpr float C2[5] = {1.0925484305920792f, -1.0925484305920792f,
        0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f};
    constexpr float C3[7] = {-0.5900435899266435f, 2.890611442640554f,
        -0.4570457994644658f, 0.3731763325901154f, -0.4570457994644658f,
        1.445305721320277f, -0.5900435899266435f};
    const float x = direction.x, y = direction.y, z = direction.z;
    glm::vec3 result = C0 * coefficients[0];
    if (order >= SphericalHarmonicsOrder::Degree1)
        result += -C1 * y * coefficients[1] + C1 * z * coefficients[2] - C1 * x * coefficients[3];
    if (order >= SphericalHarmonicsOrder::Degree2)
    {
        result += C2[0] * x * y * coefficients[4]
            + C2[1] * y * z * coefficients[5]
            + C2[2] * (2.0f * z * z - x * x - y * y) * coefficients[6]
            + C2[3] * x * z * coefficients[7]
            + C2[4] * (x * x - y * y) * coefficients[8];
    }
    if (order >= SphericalHarmonicsOrder::Degree3)
    {
        result += C3[0] * y * (3.0f * x * x - y * y) * coefficients[9]
            + C3[1] * x * y * z * coefficients[10]
            + C3[2] * y * (4.0f * z * z - x * x - y * y) * coefficients[11]
            + C3[3] * z * (2.0f * z * z - 3.0f * x * x - 3.0f * y * y) * coefficients[12]
            + C3[4] * x * (4.0f * z * z - x * x - y * y) * coefficients[13]
            + C3[5] * z * (x * x - y * y) * coefficients[14]
            + C3[6] * x * (x * x - 3.0f * y * y) * coefficients[15];
    }
    return result;
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
    state.lastBsdfPdfBits = __float_as_uint(bsdfSample.pdf);

    // Direct light sampling.
    {
        LightSample lightSample{};
        const uint32_t pl    = params.scene.pointLightCount;
        const uint32_t sl    = params.scene.spotLightCount;
        const uint32_t rl    = params.scene.rectLightCount;
        const uint32_t dl    = params.scene.directionalLightCount;
        const float analyticWeight = analyticLightSelectionWeight(params.scene);
        const float environmentWeight = bsdf.transmission <= 0.0f
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
                        position, lightRng, wl,
                        params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                        params.scene.d65);
                } else target -= w;
            }
            for (uint32_t i = 0; i < sl && selectedWeight == 0.0f; ++i) {
                const float w = params.scene.spotLights[i].selectionWeight();
                if (target < w) {
                    selectedWeight = w;
                    lightSample = params.scene.spotLights[i].sampleLi(
                        position, lightRng, wl,
                        params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                        params.scene.d65);
                } else target -= w;
            }
            for (uint32_t i = 0; i < rl && selectedWeight == 0.0f; ++i) {
                const float w = params.scene.rectLights[i].selectionWeight();
                if (target < w) {
                    selectedWeight = w;
                    lightSample = params.scene.rectLights[i].sampleLi(
                        position, lightRng, wl,
                        params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                        params.scene.d65);
                } else target -= w;
            }
            for (uint32_t i = 0; i < dl && selectedWeight == 0.0f; ++i) {
                const float w = params.scene.directionalLights[i].selectionWeight();
                if (target < w) {
                    selectedWeight = w;
                    lightSample = params.scene.directionalLights[i].sampleLi(
                        position, lightRng, wl,
                        params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                        params.scene.d65);
                } else target -= w;
            }
            if (selectedWeight == 0.0f && environmentWeight > 0.0f) {
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

NR_GPU inline void gaussianBackward(
    const KernelParams& params, uint32_t gaussianId,
    const glm::vec3& rayOrigin, const glm::vec3& rayDirection,
    const glm::vec3& dLdColor);

NR_GPU_KERNEL void shadeKernel(const KernelParams params)
{
    const uint32_t index = NR_GPU_LAUNCH_IDX;
    const uint32_t activeCount = params.queues.rayCounts[params.depth];
    if (params.train.mode == RenderMode::Backward)
    {
        if (index >= activeCount)
            return;
        const HitWorkItem hit = params.queues.hitQueue[index];
        const bool isGaussian = hit.instanceIndex != InvalidIndex
            && hit.instanceIndex >= params.scene.meshInstanceCount;
        if (!isGaussian)
            return;
        const glm::vec3 dLdImage = params.train.dLdImage[hit.sampleIndex]
            / static_cast<float>(params.train.samplesPerPixel);
        gaussianBackward(params, hit.instanceIndex - params.scene.meshInstanceCount,
            hit.rayOrigin, hit.rayDirection, dLdImage);
        return;
    }
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
        SampledWavelengths& wl = state.wl;

        // Gaussian discriminator: instanceIndex >= meshInstanceCount.
        // For gaussian hits instanceIndex encodes meshInstanceCount + gaussianId,
        // attribute0 = world-space hit distance t, attribute1 = density alpha.
        // A miss also leaves instanceIndex at InvalidIndex (0xffffffff), which
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
            if (params.train.enabled)
            {
                state.trainColor += state.cameraWeight * params.train.colorRgb[gaussianId];
                params.queues.pathStates[hit.sampleIndex] = state;
                continuePath = false;
            }
            else
            {
            const glm::vec3* shCoeffs = params.scene.gaussianShCoeffs + gaussianId * 16;
            const glm::vec3 viewDirection = glm::normalize(-hit.rayDirection);
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
            const glm::vec3 gaussianPos = hit.positionOrDirection;

            const RandomState bounceKey = state.rngState;
            state.rngState = splitMix64(bounceKey);
            RandomState bsdfRng = seedRandom(bounceKey ^ 0x13198a2e03707344ull);
            RandomState lightRng = seedRandom(bounceKey ^ 0xa4093822299f31d0ull);
            RandomState shadowRng = seedRandom(bounceKey ^ 0x082efa98ec4e6c89ull);
            RandomState rouletteRng = seedRandom(bounceKey ^ 0x452821e638d01377ull);

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
                const uint32_t pl = params.scene.pointLightCount;
                const uint32_t sl = params.scene.spotLightCount;
                const uint32_t rl = params.scene.rectLightCount;
                const uint32_t dl = params.scene.directionalLightCount;
                const float analyticWeight = analyticLightSelectionWeight(params.scene);
                const float environmentWeight = 0.0f; // env sampled by scattered ray
                const float totalWeight = analyticWeight + environmentWeight;
                if (totalWeight > 0.0f)
                {
                    float target = randomFloat(lightRng) * totalWeight;
                    float selectedWeight = 0.0f;
                    for (uint32_t i = 0; i < pl && selectedWeight == 0.0f; ++i) {
                        const float w = params.scene.pointLights[i].selectionWeight();
                        if (target < w) { selectedWeight = w;
                            lightSample = params.scene.pointLights[i].sampleLi(
                                gaussianPos, lightRng, wl,
                                params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                                params.scene.d65); } else target -= w;
                    }
                    for (uint32_t i = 0; i < sl && selectedWeight == 0.0f; ++i) {
                        const float w = params.scene.spotLights[i].selectionWeight();
                        if (target < w) { selectedWeight = w;
                            lightSample = params.scene.spotLights[i].sampleLi(
                                gaussianPos, lightRng, wl,
                                params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                                params.scene.d65); } else target -= w;
                    }
                    for (uint32_t i = 0; i < rl && selectedWeight == 0.0f; ++i) {
                        const float w = params.scene.rectLights[i].selectionWeight();
                        if (target < w) { selectedWeight = w;
                            lightSample = params.scene.rectLights[i].sampleLi(
                                gaussianPos, lightRng, wl,
                                params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                                params.scene.d65); } else target -= w;
                    }
                    for (uint32_t i = 0; i < dl && selectedWeight == 0.0f; ++i) {
                        const float w = params.scene.directionalLights[i].selectionWeight();
                        if (target < w) { selectedWeight = w;
                            lightSample = params.scene.directionalLights[i].sampleLi(
                                gaussianPos, lightRng, wl,
                                params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                                params.scene.d65); } else target -= w;
                    }
                    if (selectedWeight > 0.0f && lightSample.radiance.maxComponent() > 0.0f)
                    {
                        const float selectionPdf = selectedWeight / totalWeight;
                        const SampledSpectrum brdf = albedo * inv4Pi;
                        shadow.origin    = gaussianPos + lightSample.direction * 0.001f;
                        shadow.direction = lightSample.direction;
                        shadow.tMin      = 0.001f;
                        shadow.tMax      = lightSample.distance - 0.002f;
                        // Analytic lights are only reachable through NEE: they are
                        // not emissive geometry that the scattered ray can hit.
                        // Therefore there is no competing sampling technique and
                        // no MIS term. Compensate only for selecting one light.
                        shadow.contribution = state.throughput * brdf * lightSample.radiance
                            / fmaxf(selectionPdf, 1e-20f);
                        shadow.rngState     = shadowRng;
                        shadow.sampleIndex  = hit.sampleIndex;
                        params.queues.shadowQueue[index] = shadow;
                    }
                }
            }

            state.throughput *= albedo;
            // Gaussian scattering does not perform environment NEE. A zero PDF
            // tells the miss path not to MIS-downweight its environment sample.
            state.lastBsdfPdfBits = __float_as_uint(0.0f);

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
                            * params.scene.environment->pdf(hit.positionOrDirection);
                        misWeight = powerHeuristic(bsdfPdf, lightPdf);
                    }
                }
                state.radiance += state.throughput *
                    params.scene.environment->radiance(
                        params.scene.textures, params.scene.textureCount,
                        hit.positionOrDirection, cameraRay, wl,
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
            state.rngState = splitMix64(bounceKey);
            RandomState opacityRng = seedRandom(bounceKey ^ 0x243f6a8885a308d3ull);
            RandomState bsdfRng = seedRandom(bounceKey ^ 0x13198a2e03707344ull);
            RandomState lightRng = seedRandom(bounceKey ^ 0xa4093822299f31d0ull);
            RandomState shadowRng = seedRandom(bounceKey ^ 0x082efa98ec4e6c89ull);
            RandomState rouletteRng = seedRandom(bounceKey ^ 0x452821e638d01377ull);
            glm::vec3 nextDirection{};
            float opacity = material.opacity;
            if (material.opacityIndex >= 0)
                opacity *= params.scene.textures[material.opacityIndex].sample(surface.uv).w;
            opacity = fminf(fmaxf(opacity, 0.0f), 1.0f);
            const bool transparentSurface = randomFloat(opacityRng) > opacity;
            if (transparentSurface)
            {
                nextDirection = hit.positionOrDirection;
            }
            else
            {
                state.packedCounters |= 1u << CounterHitShift;
                if (glm::dot(shadingNormal, hit.positionOrDirection) > 0.0f)
                    shadingNormal = -shadingNormal;
                if (glm::dot(surface.geometricNormal, hit.positionOrDirection) > 0.0f)
                    surface.geometricNormal = -surface.geometricNormal;
                const glm::vec3 viewDirection = -hit.positionOrDirection;

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

// Reverse-mode Gaussian contribution for the replayed training path. The
// visibility decision is treated as a fixed stochastic sample; gradients are
// accumulated only through the continuous alpha/color evaluation.
NR_GPU inline void gaussianBackward(
    const KernelParams& params, const uint32_t gaussianId,
    const glm::vec3& rayOrigin, const glm::vec3& rayDirection,
    const glm::vec3& dLdColor)
{
    const glm::vec4 rawRotation = params.train.rotation[gaussianId];
    const float rawNorm = fmaxf(glm::length(rawRotation), 1.0e-8f);
    const glm::quat q = glm::normalize(glm::quat(
        rawRotation.w, rawRotation.x, rawRotation.y, rawRotation.z));
    const glm::mat3 rotation = glm::mat3_cast(q);
    const glm::vec3 scale = glm::exp(params.train.logScale[gaussianId]);
    const glm::vec3 relative = rayOrigin - params.train.position[gaussianId];
    const glm::vec3 objectOrigin = (glm::transpose(rotation) * relative) / scale;
    const glm::vec3 objectDirection = (glm::transpose(rotation) * rayDirection) / scale;
    const float denominator = fmaxf(glm::dot(objectDirection, objectDirection), 1.0e-8f);
    const float tClosest = -glm::dot(objectOrigin, objectDirection) / denominator;
    const glm::vec3 y = objectOrigin + tClosest * objectDirection;
    const float distanceSq = glm::dot(y, y);
    const float opacity = 1.0f / (1.0f + __expf(-params.train.opacityLogit[gaussianId]));
    const float alpha = opacity * __expf(-0.5f * distanceSq);
    const glm::vec3 yOverScale = y / scale;

    const float dLdAlpha = glm::dot(dLdColor, params.train.colorRgb[gaussianId])
        / fmaxf(alpha, 1.0e-8f);
    const float distanceScale = dLdAlpha * (-0.5f * alpha);
    const glm::vec3 dDistancePosition = -2.0f * (rotation * yOverScale);
    const glm::vec3 dDistanceScale = -2.0f * glm::vec3(y.x * y.x, y.y * y.y, y.z * y.z);

    const glm::vec3 w = relative + tClosest * rayDirection;
    const float G[3][3] = {
        {2.0f * w.x * yOverScale.x, 2.0f * w.x * yOverScale.y, 2.0f * w.x * yOverScale.z},
        {2.0f * w.y * yOverScale.x, 2.0f * w.y * yOverScale.y, 2.0f * w.y * yOverScale.z},
        {2.0f * w.z * yOverScale.x, 2.0f * w.z * yOverScale.y, 2.0f * w.z * yOverScale.z},
    };
    const float qx = q.x, qy = q.y, qz = q.z, qw = q.w;
    glm::vec4 dRotation(
        2 * G[0][1] * qy + 2 * G[0][2] * qz + 2 * G[1][0] * qy - 4 * G[1][1] * qx
            - 2 * G[1][2] * qw + 2 * G[2][0] * qz + 2 * G[2][1] * qw - 4 * G[2][2] * qx,
        -4 * G[0][0] * qy + 2 * G[0][1] * qx + 2 * G[0][2] * qw + 2 * G[1][0] * qx
            + 2 * G[1][2] * qz - 2 * G[2][0] * qw + 2 * G[2][1] * qz - 4 * G[2][2] * qy,
        -4 * G[0][0] * qz - 2 * G[0][1] * qw + 2 * G[0][2] * qx + 2 * G[1][0] * qw
            - 4 * G[1][1] * qz + 2 * G[1][2] * qy + 2 * G[2][0] * qx + 2 * G[2][1] * qy,
        -2 * G[0][1] * qz + 2 * G[0][2] * qy + 2 * G[1][0] * qz - 2 * G[1][2] * qx
            - 2 * G[2][0] * qy + 2 * G[2][1] * qx);
    const glm::vec4 qVector(q.x, q.y, q.z, q.w);
    dRotation = (dRotation - glm::dot(dRotation, qVector) * qVector) / rawNorm;

    glm::vec3* dPosition = &params.train.dPosition[gaussianId];
    const glm::vec3 positionGradient = distanceScale * dDistancePosition;
    atomicAdd(&dPosition->x, positionGradient.x);
    atomicAdd(&dPosition->y, positionGradient.y);
    atomicAdd(&dPosition->z, positionGradient.z);

    glm::vec3* dLogScale = &params.train.dLogScale[gaussianId];
    const glm::vec3 scaleGradient = distanceScale * dDistanceScale;
    atomicAdd(&dLogScale->x, scaleGradient.x);
    atomicAdd(&dLogScale->y, scaleGradient.y);
    atomicAdd(&dLogScale->z, scaleGradient.z);

    glm::vec4* dRotationOut = &params.train.dRotation[gaussianId];
    const glm::vec4 rotationGradient = distanceScale * dRotation;
    atomicAdd(&dRotationOut->x, rotationGradient.x);
    atomicAdd(&dRotationOut->y, rotationGradient.y);
    atomicAdd(&dRotationOut->z, rotationGradient.z);
    atomicAdd(&dRotationOut->w, rotationGradient.w);
    atomicAdd(&params.train.dOpacityLogit[gaussianId],
        dLdAlpha * alpha * (1.0f - opacity));
    atomicAdd(&params.train.dColorRgb[gaussianId].x, dLdColor.x);
    atomicAdd(&params.train.dColorRgb[gaussianId].y, dLdColor.y);
    atomicAdd(&params.train.dColorRgb[gaussianId].z, dLdColor.z);
}
