#pragma once

#include <optix.h>
#include <optix_device.h>

#include "Camera/Camera.h"
#include "Raytracing/Gpu/Geometry.h"
#include "Raytracing/Path/MisHeuristic.h"
#include "Raytracing/Path/ShadowTerminator.h"
#include "Samplers/OwenSobolSampler.h"
#include "Shading/CompositeBsdf.h"
#include "Shading/MaterialEvaluation.h"
#include "SVM/SvmEval.h"

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
    // reorder: whether to call optixReorder() after this traversal. SER
    // reordering has a real, roughly fixed per-call cost that is only worth
    // paying once per thread per path segment, for the ray whose hit
    // determines what (potentially divergent) material gets shaded next --
    // the main continuation ray in trace(). Shadow rays in shadowOccluded()
    // never reach a shading callable (they only ever check opacity), so
    // reordering for them earns nothing and, worse, reordering the same
    // thread a second time is not merely wasted but actively pathological:
    // measured on an RTX 4070, a single redundant intersect() call with
    // reorder enabled turned a 452ms render into 8469ms (~19x) on a trivial
    // one-sphere scene. Callers that trace more than one ray per thread per
    // segment (shadowOccluded's blocker loop) must pass false here.
    NR_GPU RayHit intersect(
        const Ray& ray,
        const float tMin,
        const float tMax,
        const uint32_t sampleIndex,
        const uint32_t excludedGaussianId = InvalidIndex,
        const bool terminateOnFirstGaussianHit = false,
        const bool reorder = true) const
    {
        RayHit hit{};
        // Any-hit is disabled for the main continuation ray's own traversal
        // (deferred CH shading below handles that material fully); shadow
        // rays (reorder == false, see this function's own comment) need it
        // enabled instead, for the SVM opacity path's cutout testing -- the only way a shadow ray, which never
        // invokes closesthit, can still respect per-material opacity.
        const unsigned int anyHitFlags =
            reorder ? OPTIX_RAY_FLAG_DISABLE_ANYHIT : OPTIX_RAY_FLAG_NONE;
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
                    tMin,
                    tMax,
                    0.0f,
                    MeshVisibility,
                    anyHitFlags,
                    0,
                    1,
                    0);
                // Deferred shading (optixHitObjectIsHit/Get*) only defers
                // choosing what to run next; without this, threads that
                // scattered onto unrelated materials and directions after
                // the first bounce stay locked to whatever lane they
                // started in, so the OSL callable below serializes across
                // every distinct material a warp's rays landed on. Bounce 0
                // stays coherent for free (adjacent pixels start off hitting
                // similar geometry); every bounce past that is where this
                // actually earns its keep. See this function's reorder
                // parameter comment for why this must not run for shadow
                // rays.
                if (reorder)
                    optixReorder();
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
                : tMax;
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
                tMin,
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
                tMin,
                tMax,
                0.0f,
                MeshVisibility,
                anyHitFlags,
                0,
                1,
                0);
            // See the mesh-hit branch above: reorders lanes by hit outcome
            // before the OSL material callable runs, which is what keeps
            // later bounces (post-scatter, no longer screen-coherent) from
            // serializing one material at a time per warp. Also see this
            // function's reorder parameter comment for why this must not
            // run for shadow rays.
            if (reorder)
                optixReorder();
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
        const glm::vec3& shadingNormal,
        float& outTMin,
        float& outTMax) const
    {
        const glm::vec3 terminatorOffset = nr::shadowTerminatorOffset(
            params.scene, surface.instanceIndex, surface.primitiveIndex,
            surface.barycentricU, surface.barycentricV,
            shadingNormal, geometricNormal, light.direction);
        const glm::vec3 toLight = light.direction * light.distance - terminatorOffset;
        const float distance = glm::length(toLight);
        if (distance <= 0.0f)
        {
            outTMin = 0.0f;
            outTMax = 0.0f;
            return Ray::invalid();
        }

        const glm::vec3 direction = toLight / distance;
        outTMin = RayOffset;
        outTMax = distance - 2.0f * RayOffset;
        return Ray(
            surface.position + terminatorOffset + geometricNormal
                * (glm::dot(direction, geometricNormal) >= 0.0f
                    ? RayOffset : -RayOffset),
            direction);
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

    NR_GPU LightHit intersectAnalyticLights(
        const Ray& ray, const float tMin, const float tMax,
        const SampledWavelengths& wavelengths) const
    {
        LightHit nearest{};
        uint32_t globalIndex = 0;
        const auto consider = [&](LightHit candidate) {
            candidate.lightIndex = globalIndex++;
            if (candidate.radiance.maxComponent() > 0.0f
                && candidate.distance <= nearest.distance)
                nearest = candidate;
        };
        for (uint32_t i = 0; i < params.scene.pointLightCount; ++i)
            consider(params.scene.pointLights[i].intersect(
                ray, tMin, tMax, wavelengths, params.scene.spectrumTableScale,
                params.scene.spectrumTableCoeffs, params.scene.d65));
        for (uint32_t i = 0; i < params.scene.spotLightCount; ++i)
            consider(params.scene.spotLights[i].intersect(
                ray, tMin, tMax, wavelengths, params.scene.spectrumTableScale,
                params.scene.spectrumTableCoeffs, params.scene.d65));
        for (uint32_t i = 0; i < params.scene.rectLightCount; ++i)
            consider(params.scene.rectLights[i].intersect(
                ray, tMin, tMax, wavelengths, params.scene.spectrumTableScale,
                params.scene.spectrumTableCoeffs, params.scene.d65));
        for (uint32_t i = 0; i < params.scene.directionalLightCount; ++i)
            consider(params.scene.directionalLights[i].intersect(
                ray, wavelengths, params.scene.spectrumTableScale,
                params.scene.spectrumTableCoeffs, params.scene.d65));
        return nearest;
    }

    NR_GPU float analyticLightHitMisWeight(
        const LightHit& light, const float bsdfPdf) const
    {
        if (bsdfPdf <= 0.0f || light.pdf <= 0.0f
            || params.scene.analyticLightAliases == nullptr
            || light.lightIndex >= params.scene.analyticLightAliasCount)
            return 1.0f;
        const float environmentWeight = fmaxf(
            params.scene.environment->importanceWeight, 0.0f);
        const float totalWeight =
            params.scene.analyticLightSelectionWeight + environmentWeight;
        if (totalWeight <= 0.0f)
            return 1.0f;
        const float selectionPdf =
            params.scene.analyticLightAliases[light.lightIndex].selectionPdf;
        const float lightPdf =
            (params.scene.analyticLightSelectionWeight / totalWeight)
            * selectionPdf * light.pdf;
        return powerHeuristic(bsdfPdf, lightPdf);
    }

    NR_GPU bool shadowOccluded(
        const Ray& ray,
        const float tMin,
        const float tMax,
        const uint32_t excludedGaussianId,
        const SampledWavelengths& wavelengths,
        RandomState& rng) const
    {
        const bool gaussian = gaussianEnabled();
        float rayMin = tMin;
        while (rayMin < tMax)
        {
            const uint32_t gaussianSampleIndex = gaussian ? randomUint(rng) : 0;
            const RayHit hit = intersect(ray, rayMin, tMax, gaussianSampleIndex,
                excludedGaussianId, true, false);
            if (hit.instanceIndex == InvalidIndex)
                return false;
            if (hit.primitiveIndex == InvalidIndex)
                return true;
            const Surface blocker = Surface::fromHit(params.scene, hit);
            const bool hasSvmProgram = blocker.material->svmBytecodeLength != 0;

            float blockProbability;
            if (!hasSvmProgram)
            {
                // No compiled SVM program: shade natively from the
                // material's plain fields instead of assuming fully opaque
                // (see NativeMaterialEvaluation.h). Native materials have no
                // any-hit program on their hitgroup (see Raytracer's
                // fallback hitgroup), so this is the only opacity test they
                // get.
                blockProbability = 1.0f;
            }
            else
            {
                MaterialEvaluation evaluation{};
                nr::shading::NoorRayCompositeBsdf bsdf(
                    blocker.geometricNormal, blocker.normal, -ray.direction());
                MaterialShadingContext shadingContext{};
                shadingContext.position = blocker.position;
                shadingContext.geometricNormal = blocker.geometricNormal;
                shadingContext.interpolatedNormal = blocker.normal;
                shadingContext.tangent = blocker.tangent;
                shadingContext.bitangent = glm::cross(blocker.normal, blocker.tangent)
                    * blocker.tangentSign;
                shadingContext.uv = blocker.uv;
                shadingContext.vertexColor = blocker.color;
                shadingContext.viewDirection = -ray.direction();
                shadingContext.primitiveId = blocker.primitiveIndex;
                const bool exiting = glm::dot(
                    -ray.direction(), blocker.geometricNormal) < 0.0f;
                if (!nr::svm::svmEvalNodes(params.scene,
                    blocker.material->svmBytecodeOffset, blocker.material->svmBytecodeLength,
                    blocker.material->svmTextureOffset, blocker.material->svmTextureCount,
                    shadingContext, wavelengths,
                    exiting, evaluation, bsdf))
                {
                    // A stale material slot is handled exactly like a native
                    // material, rather than turning the surface opaque.
                    evaluation.opacity = 1.0f;
                    bsdf.prepare();
                }
                blockProbability = fminf(fmaxf(
                    evaluation.opacity * blocker.color.a
                    * (1.0f - bsdf.transmissionEstimate()), 0.0f), 1.0f);
            }
            if (blockProbability >= 1.0f
                || randomFloat(rng) < blockProbability)
                return true;
            rayMin = hit.t + fmaxf(1e-4f, fabsf(hit.t) * 1e-6f);
        }
        return false;
    }

    // Template param works identically for any BSDF type that exposes
    // evaluate()/sample()/cosine()/shadingNormal() -- the compiler
    // instantiates and inlines a separate copy per type at no cost.
    template <typename BsdfT>
    NR_GPU SampledSpectrum estimateDirect(
        const Surface& surface,
        const glm::vec3& geometricNormal,
        const BsdfT& bsdf,
        const SampledWavelengths& wavelengths,
        RandomState& lightRng,
        RandomState& shadowRng) const
    {
        LightSample light{};
        float environmentPdf = 0.0f;
        if (!sampleDirectLight(surface.position, wavelengths,
                lightRng, light, environmentPdf))
            return SampledSpectrum(0.0f);

        float shadowTMin, shadowTMax;
        const Ray shadowRay = spawnShadowRay(
            surface, light, geometricNormal, bsdf.shadingNormal(),
            shadowTMin, shadowTMax);
        if (glm::dot(shadowRay.direction(), shadowRay.direction()) <= 0.0f
            || shadowTMax <= shadowTMin
            || shadowOccluded(shadowRay, shadowTMin, shadowTMax,
                InvalidIndex, wavelengths, shadowRng))
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
            position, light.direction, RayOffset);
        const float shadowTMin = RayOffset;
        const float shadowTMax = light.distance - 2.0f * RayOffset;
        if (glm::dot(shadowRay.direction(), shadowRay.direction()) <= 0.0f
            || shadowTMax <= shadowTMin
            || shadowOccluded(shadowRay, shadowTMin, shadowTMax,
                gaussianId, wavelengths, shadowRng))
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
        float tMin = Ray::DefaultMinDistance;
        float tMax = Ray::DefaultMaxDistance;
        for (uint32_t segment = 0; segment < params.depth; ++segment)
        {
            const uint32_t gaussianSampleIndex = gaussian
                ? hashCombine32(pixel, segment) : pixel;
            const RayHit hit = intersect(ray, tMin, tMax, gaussianSampleIndex);
            if (state.depth > 0)
            {
                const LightHit light = intersectAnalyticLights(ray, tMin, tMax, state.wl);
                const float surfaceDistance = hit.instanceIndex == InvalidIndex
                    ? Ray::InfiniteDistance : hit.t;
                if (light.radiance.maxComponent() > 0.0f
                    && light.distance <= surfaceDistance)
                {
                    state.radiance += state.throughput * light.radiance
                        * analyticLightHitMisWeight(light, state.lastBsdfPdf);
                    if (light.distance == Ray::InfiniteDistance)
                        state.radiance += state.throughput
                            * environmentRadiance(ray, state);
                    return;
                }
            }
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
                tMin = Ray::DefaultMinDistance;
                tMax = Ray::DefaultMaxDistance;
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

            MaterialEvaluation evaluation{};
            nr::shading::NoorRayCompositeBsdf bsdf(
                geometricNormal, surface.normal, -ray.direction());
            const bool exiting = glm::dot(-ray.direction(), geometricNormal) < 0.0f;
            bool svmEvaluated = false;
            if (surface.material->svmBytecodeLength != 0) {
                MaterialShadingContext shadingContext{};
                shadingContext.position = surface.position;
                shadingContext.geometricNormal = geometricNormal;
                shadingContext.interpolatedNormal = surface.normal;
                shadingContext.tangent = surface.tangent;
                shadingContext.bitangent = glm::cross(surface.normal, surface.tangent)
                    * surface.tangentSign;
                shadingContext.uv = surface.uv;
                shadingContext.vertexColor = surface.color;
                shadingContext.viewDirection = -ray.direction();
                shadingContext.primitiveId = surface.primitiveIndex;
                svmEvaluated = nr::svm::svmEvalNodes(params.scene,
                    surface.material->svmBytecodeOffset, surface.material->svmBytecodeLength,
                    surface.material->svmTextureOffset, surface.material->svmTextureCount,
                    shadingContext,
                    state.wl, exiting, evaluation, bsdf);
            }
            if (!svmEvaluated) {
                evaluation.opacity = 1.0f;
                bsdf.prepare();
            }

            const float opacity = fminf(fmaxf(
                evaluation.opacity * surface.color.a, 0.0f), 1.0f);
            if (randomFloat(randoms.opacity) > opacity)
            {
                ray = spawnSurfaceRay(surface, ray.direction(), geometricNormal);
                tMin = Ray::DefaultMinDistance;
                tMax = Ray::DefaultMaxDistance;
                continue;
            }

            const SampledSpectrum emission =
                evaluation.has(MaterialEvaluationFlags::HasEmission)
                ? rgbIlluminantToSpectrum(
                    evaluation.emission * evaluation.emissionStrength,
                    state.wl, params.scene.spectrumTableScale,
                    params.scene.spectrumTableCoeffs, params.scene.d65)
                : SampledSpectrum(0.0f);

            // Shared post-BSDF-construction shading: emission, direct lighting,
            // one scattering sample, Russian roulette, and the next ray.
            // Returns false when the path should terminate here.
            const auto shadeWithBsdf = [&](const auto& bsdf) -> bool
            {
                state.alpha = 1.0f;
                state.radiance += state.throughput * emission;
                state.radiance += state.throughput * estimateDirect(surface, geometricNormal,
                    bsdf, state.wl, randoms.light, randoms.shadow);
                const BsdfSample bsdfSample = bsdf.sample(randoms.bsdf);
                if (bsdfSample.event == BsdfEvent::Transmission)
                    state.transmit(bsdfSample.eta);
                state.scatter(bsdfSample.weight,
                    bsdfSample.singular ? 0.0f : bsdfSample.pdf);
                if (!survivesRussianRoulette(state, randoms.roulette))
                    return false;
                ray = spawnSurfaceRay(surface, bsdfSample.direction, geometricNormal);
                tMin = Ray::DefaultMinDistance;
                tMax = Ray::DefaultMaxDistance;
                return true;
            };

            if (!shadeWithBsdf(bsdf))
                return;
            continue;
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
