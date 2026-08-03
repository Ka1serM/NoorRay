#pragma once

#include <optix_device.h>

#include "Backend/OptiX/ABI/Geometry.h"
#include "Rendering/Lighting/DirectLightSampling.h"
#include "Rendering/MisHeuristic.h"
#include "Rendering/ShadowTerminator.h"
#include "Rendering/Sampling/RandomSampler.h"
#include "Materials/Shading/CompositeBsdf.h"
#include "Materials/Shading/MaterialEvaluation.h"
#include "Materials/SVM/SvmEval.h"

extern "C"
{
extern __constant__ KernelParams params;
}

namespace nr::gaussian_hit
{

static constexpr float RayOffset = Ray::DefaultMinDistance;
static constexpr float Inv4Pi = 0.07957747154594767f;
static constexpr uint32_t CoherenceHintMiss = 0;
static constexpr uint32_t CoherenceHintGaussian = 0x80u;
static constexpr uint32_t CoherenceHintMaterialBase = 0;
static constexpr uint32_t CoherenceHintBits = 8;

NR_GPU inline bool gaussianEnabled()
{
    return (params.frame.visibilityMask & GaussianVisibility) != 0;
}

NR_GPU inline uint32_t coherenceHint(const RayHit& hit)
{
    if (hit.instanceIndex == InvalidIndex)
        return CoherenceHintMiss;
    if (hit.primitiveIndex == InvalidIndex)
        return CoherenceHintGaussian;
    const GpuInstance instance = params.scene.instances[hit.instanceIndex];
    const MeshAsset& mesh = params.scene.meshes[instance.meshIndex];
    const int materialSlot = mesh.getFaces()[hit.primitiveIndex].materialIndex;
    return CoherenceHintMaterialBase
        | (mesh.getMaterialIds()[materialSlot] & 0x7Fu);
}

NR_GPU inline RayHit intersect(
    const Ray& ray,
    const float tMin,
    const float tMax,
    const uint32_t sampleIndex,
    const uint32_t excludedGaussianId = InvalidIndex,
    const bool terminateOnFirstGaussianHit = false,
    const bool reorder = true,
    const bool terminateOnFirstMeshHit = false)
{
    RayHit hit{};
    if (!ray.isValid() || !nr::isFinite(tMin) || !nr::isFinite(tMax)
        || tMin < 0.0f || tMax <= tMin)
        return hit;
    // Any-hit is disabled for traversal-only mesh queries. Beauty-path
    // traces use the separate closest-hit SBT and are launched by
    // tracePrimary() below; shadow/AOV queries continue to use this
    // hit-object path so they can inspect a hit without running shading.
    const unsigned int anyHitFlags =
        (reorder ? OPTIX_RAY_FLAG_DISABLE_ANYHIT : OPTIX_RAY_FLAG_NONE)
        | (terminateOnFirstMeshHit ? OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT : 0u);
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
                QuerySbtRecordOffset,
                1,
                0);
            // Hit-object state must be read out into meshHit before the
            // gaussian traversal below overwrites the current hit object;
            // the reorder itself is deferred to the end of this function,
            // where the gaussian result is known too and the hint can
            // describe what will actually be shaded.
            if (optixHitObjectIsHit())
            {
                const float hitT = optixHitObjectGetRayTmax();
                if (nr::isFinite(hitT) && hitT >= tMin && hitT <= tMax)
                {
                    meshHit.t = hitT;
                    meshHit.u = __uint_as_float(optixHitObjectGetAttribute_0());
                    meshHit.v = __uint_as_float(optixHitObjectGetAttribute_1());
                    meshHit.instanceIndex = optixHitObjectGetInstanceIndex();
                    meshHit.primitiveIndex = optixHitObjectGetPrimitiveIndex();
                }
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
            QuerySbtRecordOffset,
            1,
            0,
            payload0,
            payload1,
            payload2);

        const float gaussianT = __uint_as_float(payload1);
        if (nr::isFinite(gaussianT) && gaussianT >= tMin
            && gaussianT < gaussianTMax)
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
                QuerySbtRecordOffset,
                1,
                0);
        if (optixHitObjectIsHit())
        {
            const float hitT = optixHitObjectGetRayTmax();
            if (nr::isFinite(hitT) && hitT >= tMin && hitT <= tMax)
            {
                hit.t = hitT;
                hit.u = __uint_as_float(optixHitObjectGetAttribute_0());
                hit.v = __uint_as_float(optixHitObjectGetAttribute_1());
                hit.instanceIndex = optixHitObjectGetInstanceIndex();
                hit.primitiveIndex = optixHitObjectGetPrimitiveIndex();
            }
        }
    }
    // Reorder once, here, where the full hit is known: threads that
    // scattered onto unrelated materials and directions after the first
    // bounce would otherwise stay locked to whatever lane they started
    // in, serializing the SVM interpreter across every distinct material
    // a warp's rays landed on. Bounce 0 stays coherent for free (adjacent
    // pixels start off hitting similar geometry); every bounce past that
    // is where this earns its keep. See this function's reorder parameter
    // comment for why this must not run for shadow rays. hit is held in
    // thread state, which SER migrates with the thread.
    if (reorder)
        optixReorder(coherenceHint(hit), CoherenceHintBits);
    return hit;
}

template <typename Rng>
NR_GPU inline glm::vec3 sampleIsotropicDirection(Rng& rng)
{
    const float theta = 6.28318530717958647692f * randomFloat(rng);
    const float z = 2.0f * randomFloat(rng) - 1.0f;
    const float radius = sqrtf(fmaxf(1.0f - z * z, 0.0f));
    float sinTheta = 0.0f;
    float cosTheta = 1.0f;
    sincosf(theta, &sinTheta, &cosTheta);
    return glm::vec3(radius * cosTheta, z, radius * sinTheta);
}

template <typename Rng>
NR_GPU inline bool sampleDirectLight(
    const glm::vec3& position,
    const SampledWavelengths& wavelengths,
    Rng& rng,
    LightSample& light,
    float& analyticPdf,
    float& environmentPdf)
{
    return nr::direct_light::sampleDirectLight(
        position, wavelengths, rng, light, analyticPdf, environmentPdf);
}

NR_GPU inline LightHit intersectAnalyticLights(
    const Ray& ray, const float tMin, const float tMax,
    const float surfaceDistance,
    const SampledWavelengths& wavelengths)
{
    LightHit nearest{};
    const auto consider = [&](LightHit candidate, const uint32_t candidateIndex) {
        candidate.lightIndex = candidateIndex;
        if (candidate.radiance.maxComponent() > 0.0f
            && candidate.distance <= nearest.distance)
            nearest = candidate;
    };
    // Finite analytic lights are now terminal hits in the separate light GAS.
    // Only distant lights have no finite geometry and remain a fallback here.
    if (surfaceDistance == Ray::InfiniteDistance)
        for (uint32_t i = 0; i < params.scene.directionalLightCount; ++i)
        {
            const uint32_t candidateIndex =
                params.scene.directionalLightCandidateOffset + i;
            if (params.scene.directLightCandidates != nullptr
                && candidateIndex < params.scene.directLightCandidateCount)
                consider(nr::direct_light::intersectDirectionalCandidate(
                    params.scene.directLightCandidates[candidateIndex],
                    ray, wavelengths), candidateIndex);
        }
    return nearest;
}

NR_GPU inline float analyticLightHitMisWeightAt(
    const LightHit& light, const glm::vec3 position, const float bsdfPdf)
{
    (void)position;
    if (bsdfPdf <= 0.0f || light.pdf <= 0.0f)
        return 1.0f;
    return powerHeuristic(bsdfPdf,
        nr::direct_light::lightPdf(light.lightIndex, light.pdf));
}

NR_GPU inline SampledSpectrum environmentRadiance(
    const Ray& ray, const PathState& state)
{
    const bool cameraRay = state.depth == 0;
    float misWeight = 1.0f;
    if (!cameraRay)
    {
        const float environmentWeight = fmaxf(
            params.scene.environment->importanceWeight, 0.0f);
        if (state.lastBsdfPdf > 0.0f && environmentWeight > 0.0f
            && nr::direct_light::environmentLightMixtureProbability() > 0.0f)
        {
            misWeight = powerHeuristic(state.lastBsdfPdf,
                nr::direct_light::lightPdf(
                    params.scene.environment->pdf(ray.direction())));
        }
    }
    return params.scene.environment->radiance(
        params.scene.textures, params.scene.textureCount, ray.direction(), cameraRay,
        state.wl, params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
        params.scene.d65) * misWeight;
}

template <typename Rng>
NR_GPU inline bool shadowOccluded(
    const Ray& ray,
    const float tMin,
    const float tMax,
    const uint32_t excludedGaussianId,
    const SampledWavelengths& wavelengths,
    Rng& rng)
{
    if (!ray.isValid() || !nr::isFinite(tMin) || !nr::isFinite(tMax)
        || tMin < 0.0f || tMax <= tMin)
        return false;
    const bool gaussian = gaussianEnabled()
        && params.scene.renderSettings.gaussianShadingMode
            != GaussianShadingMode::DirectColor;
    if (!gaussian && params.scene.allMaterialsOpaque != 0u)
    {
        const RayHit hit = intersect(ray, tMin, tMax, 0u,
            InvalidIndex, false, false, true);
        return hit.instanceIndex != InvalidIndex;
    }
    float rayMin = tMin;
    for (uint32_t iteration = 0; iteration < 128 && rayMin < tMax; ++iteration)
    {
        const uint32_t gaussianSampleIndex = gaussian ? randomUint(rng) : 0;
        const RayHit hit = intersect(ray, rayMin, tMax, gaussianSampleIndex,
            excludedGaussianId, true, false);
        if (hit.instanceIndex == InvalidIndex)
            return false;
        if (hit.primitiveIndex == InvalidIndex)
            return true;
        const Surface blocker = Surface::fromHit(params.scene, hit);
        if (blocker.material->shadowOpaque != 0u
            && blocker.color.a >= 1.0f)
            return true;
        const bool hasSvmProgram = blocker.material->svmBytecodeLength != 0;

        float blockProbability;
        if (!hasSvmProgram)
        {
            // Every material is expected to have an SVM program by the time
            // it reaches the GPU. A missing program is treated as opaque
            // while the material compiler repairs the slot.
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
            const GpuInstance blockerInstance = params.scene.instances[blocker.instanceIndex];
            shadingContext.objectToWorld = blockerInstance.objectToWorld;
            shadingContext.worldToObject = blockerInstance.worldToObject;
            shadingContext.normalToWorld = blockerInstance.normalToWorld;
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
        const float nextRayMin = hit.t
            + fmaxf(1e-4f, fabsf(hit.t) * 1e-6f);
        if (!nr::isFinite(nextRayMin) || nextRayMin <= rayMin)
            return true;
        rayMin = nextRayMin;
    }
    return rayMin < tMax;
}

template <typename Rng>
NR_GPU inline SampledSpectrum estimateDirect(
    const glm::vec3& position,
    const uint32_t gaussianId,
    const SampledSpectrum& albedo,
    const SampledWavelengths& wavelengths,
    Rng& lightRng,
    Rng& shadowRng)
{
    LightSample light{};
    float analyticPdf = 0.0f;
    float environmentPdf = 0.0f;
    if (!sampleDirectLight(position, wavelengths,
            lightRng, light, analyticPdf, environmentPdf))
        return SampledSpectrum(0.0f);
    if (!nr::isFinite(light.direction) || !nr::isFinite(light.distance)
        || !nr::isFinite(analyticPdf) || !nr::isFinite(environmentPdf)
        || !light.radiance.isFinite())
        return SampledSpectrum(0.0f);

    const Ray shadowRay = Ray::fromOffset(
        position, light.direction, RayOffset);
    const float shadowTMin = RayOffset;
    const float shadowTMax = light.distance - 2.0f * RayOffset;
    if (!shadowRay.isValid() || !nr::isFinite(shadowTMin)
        || !nr::isFinite(shadowTMax)
        || shadowTMax <= shadowTMin
        || shadowOccluded(shadowRay, shadowTMin, shadowTMax,
            gaussianId, wavelengths, shadowRng))
        return SampledSpectrum(0.0f);
    if (environmentPdf > 0.0f)
        light.radiance *= powerHeuristic(environmentPdf, Inv4Pi)
            / fmaxf(environmentPdf, 1.0e-20f);
    if (analyticPdf > 0.0f)
        light.radiance *= powerHeuristic(analyticPdf, Inv4Pi);
    const SampledSpectrum result = albedo * Inv4Pi * light.radiance;
    return result.isFinite() ? result : SampledSpectrum(0.0f);
}

NR_GPU inline SampledSpectrum gaussianAlbedo(
    const glm::vec3 rgb,
    const SampledWavelengths& wavelengths)
{
    return rgbAlbedoToSpectrumTexture(rgb, wavelengths,
        params.scene.spectrumTableScale, params.scene.spectrumTableTexture);
}

NR_GPU inline SampledSpectrum gaussianDirectRadiance(
    const glm::vec3 rgb,
    const SampledWavelengths& wavelengths)
{
    return rgbIlluminantToSpectrumTexture(rgb, wavelengths,
        params.scene.spectrumTableScale, params.scene.spectrumTableTexture,
        params.scene.d65);
}

template <typename Rng>
NR_GPU inline bool survivesRussianRoulette(PathState& state, Rng& rng)
{
    if (!state.throughput.isFinite() || !nr::isFinite(state.etaScale))
        return false;
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

NR_GPU inline bool terminateForAnalyticLight(
    PathTracePayload& payload, const float surfaceDistance)
{
    PathState& state = *payload.state;
    if (state.depth == 0)
        return false;
    const LightHit light = intersectAnalyticLights(
        payload.ray, Ray::DefaultMinDistance, Ray::DefaultMaxDistance,
        surfaceDistance, state.wl);
    if (light.radiance.maxComponent() <= 0.0f
        || light.distance > surfaceDistance)
        return false;
    state.radiance += state.throughput * light.radiance
        * analyticLightHitMisWeightAt(
            light, payload.ray.origin(), state.lastBsdfPdf);
    // A distant/sun analytic hit is terminal, just like Cycles' light
    // intersection shader.  Do not also evaluate the world on this same ray:
    // that double-counts the continuation contribution in sun + world scenes.
    payload.status = PathTraceStatus::Terminate;
    return true;
}

NR_GPU inline void handleGaussianClosestHit(
    PathTracePayload& payload, const uint32_t gaussianId,
    const float hitT)
{
    PathState& state = *payload.state;
    if (!payload.ray.isValid() || !nr::isFinite(hitT)
        || hitT < Ray::DefaultMinDistance
        || hitT > Ray::DefaultMaxDistance)
    {
        payload.status = PathTraceStatus::Terminate;
        return;
    }
    if (terminateForAnalyticLight(payload, hitT))
        return;

    const glm::vec3 gaussianRgb = gaussianAlbedoRgb(
        params.scene.gaussianShCoeffs,
        params.scene.gaussianShCoefficientCount,
        gaussianId,
        params.scene.renderSettings.gaussianRenderSphericalHarmonics,
        -payload.ray.direction());
    if (!nr::isFinite(gaussianRgb))
    {
        payload.status = PathTraceStatus::Terminate;
        return;
    }

    if (params.scene.renderSettings.gaussianShadingMode
        == GaussianShadingMode::DirectColor)
    {
        state.radiance += state.throughput
            * gaussianDirectRadiance(gaussianRgb, state.wl);
        state.alpha = 1.0f;
        payload.status = PathTraceStatus::Terminate;
        return;
    }

    const glm::vec3 position = payload.ray.at(hitT);
    if (!nr::isFinite(position))
    {
        payload.status = PathTraceStatus::Terminate;
        return;
    }

    const SampledSpectrum albedo = gaussianAlbedo(gaussianRgb, state.wl);
    if (!albedo.isFinite())
    {
        payload.status = PathTraceStatus::Terminate;
        return;
    }
    const bool includeRoulette = static_cast<int>(state.depth + 1u)
        >= params.scene.renderSettings.russianRouletteStartBounce;
    PathRandomStreams randoms = state.nextRandomStreams(
        payload.pixel, params.frame.totalAccumulated, false, includeRoulette);
    state.radiance += state.throughput * estimateDirect(
        position, gaussianId, albedo, state.wl,
        randoms.light, randoms.shadow);
    const glm::vec3 direction = sampleIsotropicDirection(randoms.bsdf);
    payload.ray = Ray::fromOffset(position, direction, RayOffset);
    state.scatter(albedo, Inv4Pi);
    state.alpha = 1.0f;
    payload.status = survivesRussianRoulette(state, randoms.roulette)
        ? PathTraceStatus::Continue : PathTraceStatus::Terminate;
}


}

extern "C" __global__ void __closesthit__gaussian()
{
    PathTracePayload* const payload = getPathTracePayload<>();
    const uint32_t localGaussianId = optixGetInstanceIndex();
    const uint32_t* const instanceOffsets = params.scene.gaussianInstanceOffsets;
    const uint32_t globalGaussianId = instanceOffsets == nullptr
        ? localGaussianId
        : instanceOffsets[optixGetInstanceIdFromHandle(
            optixGetTransformListHandle(0))] + localGaussianId;
    nr::gaussian_hit::handleGaussianClosestHit(
        *payload, globalGaussianId, optixGetRayTmax());
}
