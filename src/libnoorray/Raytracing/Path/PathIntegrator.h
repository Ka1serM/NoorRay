#pragma once

#include <optix.h>
#include <optix_device.h>

#include "Camera/Camera.h"
#include "Raytracing/Gpu/Geometry.h"
#include "Raytracing/Path/MisHeuristic.h"
#include "Raytracing/Path/ShadowTerminator.h"
#include "Samplers/OwenSobolSampler.h"
#include "Shading/Bsdf.h"

// PathIntegrator owns transport policy: path continuation, direct-light
// selection, visibility, MIS, and termination. Persistent domain objects own
// their local operations (camera projection, material/BSDF evaluation, light
// sampling, environment evaluation, and sensor accumulation).
class PathIntegrator
{
public:
    NR_GPU explicit PathIntegrator(const KernelParams& params_) : params(params_) {}

    NR_GPU void renderSample(
        const uint32_t pixel, const uint32_t x, const uint32_t y) const
    {
        const OwenSobolSampler sampler({
            params.frame.totalAccumulated, hashCombine32(x, y)});
        SampledWavelengths wavelengths = SampledWavelengths::sampleVisible(
            sampler.sample1D(SampleDimension::Wavelength));
        glm::vec2 jitter(0.5f);
        if (params.frame.frameIndex != 0)
            jitter = sampler.sample2D(PixelSampleDimensions);
        const glm::vec2 lensSample = sampler.sample2D(LensSampleDimensions);
        const float nx = (static_cast<float>(x) + jitter.x)
            / static_cast<float>(params.frame.width) * 2.0f - 1.0f;
        const float ny = 1.0f - (static_cast<float>(y) + jitter.y)
            / static_cast<float>(params.frame.height) * 2.0f;
        const nr::rstd::optional<CameraSample> cameraSample = params.scene.camera->Dispatch(
            [&](const auto* camera) {
                const float filmY =
                    camera->getSensor().origin() == SensorOrigin::LowerLeft
                    ? -ny : ny;
                return camera->generateRay(
                    nx, filmY, lensSample, pixel, wavelengths);
            });

        PathState state{};
        state.wl = wavelengths;
        if (cameraSample)
        {
            state.throughput = SampledSpectrum(cameraSample->weight);
            state.etaScale = 1.0f;
            trace(cameraSample->ray, state, pixel);
        }
        else
            state.alpha = params.scene.camera->Dispatch(
                [](const auto* camera) { return camera->invalidRayIsOpaque() ? 1.0f : 0.0f; });
        writeSensorSample(pixel, x, y, state);
    }

    // The integrator owns the query needed by its transport estimators. AOVs
    // reuse this exact query rather than giving the scene a renderer-specific API.
    NR_GPU RayHit intersect(
        const Ray& ray,
        const uint32_t sampleIndex,
        const uint32_t excludedGaussianId = InvalidIndex,
        const bool terminateOnFirstGaussianHit = false) const
    {
        RayHit hit{};
        const bool gaussian = gaussianEnabled();
        if (gaussian)
        {
            RayHit meshHit{};
            if (params.scene.meshInstanceCount > 0)
            {
                optixTraverse(
                    params.scene.tlasHandle,
                    make_float3(ray.origin().x, ray.origin().y, ray.origin().z),
                    make_float3(ray.direction().x, ray.direction().y, ray.direction().z),
                    ray.minDistance(),
                    ray.maxDistance(),
                    0.0f,
                    MeshVisibility,
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                    0,
                    1,
                    0);
                if (optixHitObjectIsHit())
                {
                    meshHit.t = optixHitObjectGetRayTmax();
                    meshHit.u = __uint_as_float(optixHitObjectGetAttribute_0());
                    meshHit.v = __uint_as_float(optixHitObjectGetAttribute_1());
                    meshHit.instanceIndex = optixHitObjectGetInstanceIndex();
                    meshHit.primitiveIndex = optixHitObjectGetPrimitiveIndex();
                }
            }

            const float gaussianTMax = meshHit.instanceIndex != InvalidIndex
                ? meshHit.t
                : ray.maxDistance();
            uint32_t payload0 = sampleIndex;
            uint32_t payload1 = __float_as_uint(gaussianTMax);
            uint32_t payload2 = excludedGaussianId;
            const unsigned int gaussianRayFlags = OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES
                | (terminateOnFirstGaussianHit
                    ? OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                    : OPTIX_RAY_FLAG_NONE);
            optixTraverse(
                params.scene.tlasHandle,
                make_float3(ray.origin().x, ray.origin().y, ray.origin().z),
                make_float3(ray.direction().x, ray.direction().y, ray.direction().z),
                ray.minDistance(),
                gaussianTMax,
                0.0f,
                GaussianVisibility,
                gaussianRayFlags,
                0,
                1,
                0,
                payload0,
                payload1,
                payload2);

            const float gaussianT = __uint_as_float(payload1);
            if (gaussianT < gaussianTMax)
            {
                hit.instanceIndex = payload2;
                hit.t = gaussianT;
            }
            else if (meshHit.instanceIndex != InvalidIndex)
            {
                hit = meshHit;
            }
        }
        else
        {
            optixTraverse(
                params.scene.tlasHandle,
                make_float3(ray.origin().x, ray.origin().y, ray.origin().z),
                make_float3(ray.direction().x, ray.direction().y, ray.direction().z),
                ray.minDistance(),
                ray.maxDistance(),
                0.0f,
                MeshVisibility,
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                0,
                1,
                0);
            if (optixHitObjectIsHit())
            {
                hit.t = optixHitObjectGetRayTmax();
                hit.u = __uint_as_float(optixHitObjectGetAttribute_0());
                hit.v = __uint_as_float(optixHitObjectGetAttribute_1());
                hit.instanceIndex = optixHitObjectGetInstanceIndex();
                hit.primitiveIndex = optixHitObjectGetPrimitiveIndex();
            }
        }
        return hit;
    }

private:
    static constexpr float RayOffset = Ray::DefaultMinDistance;
    static constexpr float Inv4Pi = 0.07957747154594767f;

    NR_GPU bool gaussianEnabled() const
    {
        return (params.frame.visibilityMask & GaussianVisibility) != 0;
    }

    NR_GPU static glm::vec3 orientedNormal(const Surface& surface, const Ray& ray)
    {
        return glm::dot(surface.geometricNormal, ray.direction()) > 0.0f
            ? -surface.geometricNormal : surface.geometricNormal;
    }

    NR_GPU static Ray spawnSurfaceRay(
        const Surface& surface,
        const glm::vec3& direction,
        const glm::vec3& normal)
    {
        return Ray(surface.position + normal
                * (glm::dot(direction, normal) >= 0.0f ? RayOffset : -RayOffset),
            direction);
    }

    NR_GPU Ray spawnShadowRay(
        const Surface& surface,
        const LightSample& light,
        const glm::vec3& geometricNormal,
        const glm::vec3& shadingNormal) const
    {
        const glm::vec3 terminatorOffset = nr::shadowTerminatorOffset(
            params.scene, surface.instanceIndex, surface.primitiveIndex,
            surface.barycentricU, surface.barycentricV,
            shadingNormal, geometricNormal, light.direction);
        const glm::vec3 toLight = light.direction * light.distance - terminatorOffset;
        const float distance = glm::length(toLight);
        if (distance <= 0.0f)
            return Ray::invalid();

        const glm::vec3 direction = toLight / distance;
        return Ray(
            surface.position + terminatorOffset + geometricNormal
                * (glm::dot(direction, geometricNormal) >= 0.0f
                    ? RayOffset : -RayOffset),
            direction,
            RayOffset,
            distance - 2.0f * RayOffset);
    }

    NR_GPU static glm::vec3 sampleIsotropicDirection(RandomState& rng)
    {
        const float theta = 6.28318530717958647692f * randomFloat(rng);
        const float z = 2.0f * randomFloat(rng) - 1.0f;
        const float radius = sqrtf(fmaxf(1.0f - z * z, 0.0f));
        float sinTheta = 0.0f;
        float cosTheta = 1.0f;
        sincosf(theta, &sinTheta, &cosTheta);
        return glm::vec3(radius * cosTheta, z, radius * sinTheta);
    }

    // Gaussian SH colours encode outgoing RGB radiance in DirectColor mode,
    // but act as reflectance in the globally illuminated mode. These are
    // distinct RGB-to-spectrum operations: a reflectance spectrum only
    // reproduces its RGB value after multiplication by an illuminant.
    NR_GPU SampledSpectrum gaussianAlbedo(
        const glm::vec3 rgb,
        const SampledWavelengths& wavelengths) const
    {
        return rgbAlbedoToSpectrumTexture(rgb, wavelengths,
            params.scene.spectrumTableScale, params.scene.spectrumTableTexture);
    }

    NR_GPU SampledSpectrum gaussianDirectRadiance(
        const glm::vec3 rgb,
        const SampledWavelengths& wavelengths) const
    {
        return rgbIlluminantToSpectrumTexture(rgb, wavelengths,
            params.scene.spectrumTableScale, params.scene.spectrumTableTexture,
            params.scene.d65);
    }

    NR_GPU bool sampleAnalyticLight(
        const glm::vec3& position,
        const SampledWavelengths& wavelengths,
        RandomState& rng,
        LightSample& light,
        float& selectionPdf) const
    {
        selectionPdf = 0.0f;
        if (params.scene.analyticLightAliasCount == 0
            || params.scene.analyticLightAliases == nullptr)
            return false;

        const float tableSample = randomFloat(rng) * params.scene.analyticLightAliasCount;
        const uint32_t column = min(static_cast<uint32_t>(tableSample),
            params.scene.analyticLightAliasCount - 1);
        const AnalyticLightAliasEntry columnEntry =
            params.scene.analyticLightAliases[column];
        const uint32_t selected = tableSample - static_cast<float>(column)
                < columnEntry.threshold
            ? column
            : columnEntry.alias;
        selectionPdf = params.scene.analyticLightAliases[selected].selectionPdf;

        uint32_t localIndex = selected;
        if (localIndex < params.scene.pointLightCount)
        {
            light = params.scene.pointLights[localIndex].sampleLi(position, rng, wavelengths,
                params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                params.scene.d65);
            return true;
        }
        localIndex -= params.scene.pointLightCount;
        if (localIndex < params.scene.spotLightCount)
        {
            light = params.scene.spotLights[localIndex].sampleLi(position, rng, wavelengths,
                params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                params.scene.d65);
            return true;
        }
        localIndex -= params.scene.spotLightCount;
        if (localIndex < params.scene.rectLightCount)
        {
            light = params.scene.rectLights[localIndex].sampleLi(position, rng, wavelengths,
                params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
                params.scene.d65);
            return true;
        }
        localIndex -= params.scene.rectLightCount;
        if (localIndex < params.scene.directionalLightCount)
        {
            light = params.scene.directionalLights[localIndex].sampleLi(position, rng,
                wavelengths, params.scene.spectrumTableScale,
                params.scene.spectrumTableCoeffs, params.scene.d65);
            return true;
        }
        return false;
    }

    NR_GPU bool sampleDirectLight(
        const glm::vec3& position,
        const SampledWavelengths& wavelengths,
        RandomState& rng,
        LightSample& light,
        float& environmentPdf) const
    {
        environmentPdf = 0.0f;
        const float analyticWeight = params.scene.analyticLightSelectionWeight;
        const float environmentWeight = fmaxf(
            params.scene.environment->importanceWeight, 0.0f);
        const float totalWeight = analyticWeight + environmentWeight;
        if (totalWeight <= 0.0f)
            return false;

        if (randomFloat(rng) * totalWeight < analyticWeight && analyticWeight > 0.0f)
        {
            float selectionPdf = 0.0f;
            if (!sampleAnalyticLight(position, wavelengths, rng, light, selectionPdf)
                || selectionPdf <= 0.0f)
                return false;
            light.radiance *= totalWeight / (selectionPdf * analyticWeight);
            return light.radiance.maxComponent() > 0.0f;
        }

        if (environmentWeight <= 0.0f)
            return false;
        const EnvironmentSample sample = params.scene.environment->sampleDirection(rng);
        if (sample.pdf <= 0.0f)
            return false;
        light.direction = sample.direction;
        light.distance = 1e16f;
        light.radiance = params.scene.environment->radiance(
            params.scene.textures, params.scene.textureCount, sample.direction, false,
            wavelengths, params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
            params.scene.d65);
        environmentPdf = (environmentWeight / totalWeight) * sample.pdf;
        return light.radiance.maxComponent() > 0.0f;
    }

    NR_GPU bool shadowOccluded(
        const Ray& ray,
        const uint32_t excludedGaussianId,
        RandomState& rng) const
    {
        const bool gaussian = gaussianEnabled();
        float rayMin = ray.minDistance();
        while (rayMin < ray.maxDistance())
        {
            const uint32_t gaussianSampleIndex = gaussian ? randomUint(rng) : 0;
            const RayHit hit = intersect(ray.withMinDistance(rayMin), gaussianSampleIndex,
                excludedGaussianId, true);
            if (hit.instanceIndex == InvalidIndex)
                return false;
            if (hit.primitiveIndex == InvalidIndex)
                return true;
            const ShadowSurface blocker = ShadowSurface::fromHit(params.scene, hit);
            if (blocker.material->blocksShadowRay(params.scene.textures, blocker.uv, rng))
                return true;
            rayMin = hit.t + fmaxf(1e-4f, fabsf(hit.t) * 1e-6f);
        }
        return false;
    }

    NR_GPU SampledSpectrum estimateDirect(
        const Surface& surface,
        const glm::vec3& geometricNormal,
        const Bsdf& bsdf,
        const SampledWavelengths& wavelengths,
        RandomState& lightRng,
        RandomState& shadowRng) const
    {
        LightSample light{};
        float environmentPdf = 0.0f;
        if (!sampleDirectLight(surface.position, wavelengths,
                lightRng, light, environmentPdf))
            return SampledSpectrum(0.0f);

        const Ray shadowRay = spawnShadowRay(
            surface, light, geometricNormal, bsdf.shadingNormal());
        if (!shadowRay.hasTraversalInterval()
            || shadowOccluded(shadowRay, InvalidIndex, shadowRng))
            return SampledSpectrum(0.0f);
        const BsdfEvaluation evaluation = bsdf.evaluate(light.direction);
        if (environmentPdf > 0.0f)
            light.radiance *= powerHeuristic(
                environmentPdf, evaluation.pdf)
                / fmaxf(environmentPdf, 1.0e-20f);
        return evaluation.value * light.radiance
            * bsdf.cosine(light.direction);
    }

    NR_GPU SampledSpectrum estimateDirect(
        const glm::vec3& position,
        const uint32_t gaussianId,
        const SampledSpectrum& albedo,
        const SampledWavelengths& wavelengths,
        RandomState& lightRng,
        RandomState& shadowRng) const
    {
        LightSample light{};
        float environmentPdf = 0.0f;
        if (!sampleDirectLight(position, wavelengths,
                lightRng, light, environmentPdf))
            return SampledSpectrum(0.0f);

        const Ray shadowRay = Ray::fromOffset(
            position, light.direction, RayOffset, RayOffset,
            light.distance - 2.0f * RayOffset);
        if (!shadowRay.hasTraversalInterval()
            || shadowOccluded(shadowRay, gaussianId, shadowRng))
            return SampledSpectrum(0.0f);
        if (environmentPdf > 0.0f)
            light.radiance *= powerHeuristic(environmentPdf, Inv4Pi)
                / fmaxf(environmentPdf, 1.0e-20f);
        return albedo * Inv4Pi * light.radiance;
    }

    NR_GPU SampledSpectrum environmentRadiance(
        const Ray& ray, const PathState& state) const
    {
        const bool cameraRay = state.depth == 0;
        float misWeight = 1.0f;
        if (!cameraRay)
        {
            const float environmentWeight = fmaxf(
                params.scene.environment->importanceWeight, 0.0f);
            const float totalWeight = params.scene.analyticLightSelectionWeight
                + environmentWeight;
            if (state.lastBsdfPdf > 0.0f && environmentWeight > 0.0f
                && totalWeight > 0.0f)
            {
                const float lightPdf = (environmentWeight / totalWeight)
                    * params.scene.environment->pdf(ray.direction());
                misWeight = powerHeuristic(state.lastBsdfPdf, lightPdf);
            }
        }
        return params.scene.environment->radiance(
            params.scene.textures, params.scene.textureCount, ray.direction(), cameraRay,
            state.wl, params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
            params.scene.d65) * misWeight;
    }

    NR_GPU bool survivesRussianRoulette(PathState& state, RandomState& rng) const
    {
        if (static_cast<int>(state.depth)
            < params.scene.renderSettings.russianRouletteStartBounce)
            return true;
        const float survival = fminf(fmaxf(
            state.throughput.maxComponent() * state.etaScale, 0.05f), 0.95f);
        if (randomFloat(rng) > survival)
            return false;
        state.throughput *= 1.0f / survival;
        return true;
    }

    NR_GPU void trace(Ray ray, PathState& state, const uint32_t pixel) const
    {
        const bool gaussian = gaussianEnabled();
        for (uint32_t segment = 0; segment < params.depth; ++segment)
        {
            const uint32_t gaussianSampleIndex = gaussian
                ? hashCombine32(pixel, segment) : pixel;
            const RayHit hit = intersect(ray, gaussianSampleIndex);
            if (hit.instanceIndex == InvalidIndex)
            {
                const bool cameraPath = state.depth == 0;
                const bool backgroundVisible =
                    !params.scene.renderSettings.transparentBackground;
                if (!cameraPath || backgroundVisible)
                    state.radiance += state.throughput * environmentRadiance(ray, state);
                if (cameraPath && backgroundVisible)
                    state.alpha = 1.0f;
                return;
            }

            if (hit.primitiveIndex == InvalidIndex)
            {
                const glm::vec3 position = ray.at(hit.t);
                const glm::vec3 gaussianRgb = gaussianAlbedoRgb(
                    params.scene.gaussianShCoeffs,
                    params.scene.gaussianShCoefficientCount,
                    hit.instanceIndex,
                    params.scene.renderSettings.gaussianRenderSphericalHarmonics,
                    -ray.direction());
                if (params.scene.renderSettings.gaussianShadingMode
                    == GaussianShadingMode::DirectColor)
                {
                    state.radiance += state.throughput * gaussianDirectRadiance(
                        gaussianRgb, state.wl);
                    state.alpha = 1.0f;
                    return;
                }

                const SampledSpectrum albedo = gaussianAlbedo(gaussianRgb, state.wl);

                PathRandomStreams randoms = state.nextRandomStreams(
                    pixel, params.frame.totalAccumulated);
                state.radiance += state.throughput * estimateDirect(position,
                    hit.instanceIndex, albedo, state.wl, randoms.light, randoms.shadow);
                const glm::vec3 direction = sampleIsotropicDirection(randoms.bsdf);
                ray = Ray::fromOffset(position, direction, RayOffset);
                state.scatter(albedo, Inv4Pi);
                state.alpha = 1.0f;
                if (!survivesRussianRoulette(state, randoms.roulette))
                    return;
                continue;
            }

            const Surface surface = Surface::fromHit(params.scene, hit);
            PathRandomStreams randoms = state.nextRandomStreams(
                pixel, params.frame.totalAccumulated);
            const glm::vec3 geometricNormal = orientedNormal(surface, ray);
            if (!surface.material->acceptsRayHit(
                    params.scene.textures, surface.uv, randoms.opacity))
            {
                ray = spawnSurfaceRay(surface, ray.direction(), geometricNormal);
                continue;
            }

            // A dispersive BSDF has wavelength-dependent branch probabilities,
            // refractive half-vectors, Jacobians, and PDFs. Collapse the packet
            // before evaluating direct lighting so every quantity belongs to the
            // same hero wavelength. Constant-IOR glass can retain the packet.
            if (surface.material->sampleTransmission(
                    params.scene.textures, surface.uv) > 0.0f
                && surface.material->hasDispersiveIor(state.wl))
                state.wl.terminateSecondary();

            const Bsdf bsdf(*surface.material, surface, params.scene,
                ray, state.wl);
            state.alpha = 1.0f;
            state.radiance += state.throughput * surface.material->emissionSpectral(
                params.scene.textures, surface.uv, state.wl, params.scene.spectrumTableScale,
                params.scene.spectrumTableCoeffs, params.scene.d65);
            state.radiance += state.throughput * estimateDirect(surface, geometricNormal,
                bsdf, state.wl, randoms.light, randoms.shadow);
            const BsdfSample bsdfSample = bsdf.sample(randoms.bsdf);
            if (bsdfSample.event == BsdfEvent::Transmission)
                state.transmit(bsdfSample.eta);
            state.scatter(bsdfSample.weight,
                bsdfSample.singular ? 0.0f : bsdfSample.pdf);
            if (!survivesRussianRoulette(state, randoms.roulette))
                return;
            ray = spawnSurfaceRay(surface, bsdfSample.direction, geometricNormal);
        }
    }

    NR_GPU void writeSensorSample(
        const uint32_t pixel,
        const uint32_t x,
        const uint32_t y,
        const PathState& state) const
    {
        SensorSampleContext ctx{};
        ctx.accumulation = params.accumulation;
        ctx.noiseMoments = params.noiseMoments;
        ctx.psfBuckets = params.psfGatherBuckets;
        ctx.width = params.frame.width;
        ctx.height = params.frame.height;
        ctx.totalAccumulated = params.frame.totalAccumulated;
        ctx.alpha = state.alpha;
        ctx.cieX = params.scene.cieX;
        ctx.cieY = params.scene.cieY;
        ctx.cieZ = params.scene.cieZ;
        const Sensor& sensor = params.scene.camera->Dispatch(
            [](const auto* camera) -> const Sensor& { return camera->getSensor(); });
        sensor.Dispatch([&](const auto* concreteSensor) {
            concreteSensor->addSample(pixel, state.radiance, state.wl, 1.0f, ctx,
                [&](const glm::vec4 out) {
                    surf2Dwrite(make_float4(out.x, out.y, out.z, out.w),
                        params.output.color, x * sizeof(float4), y);
                });
        });
    }

    const KernelParams& params;
};
