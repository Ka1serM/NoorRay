#pragma once

#include "Backend/OptiX/ABI/Geometry.h"
#include "Scene/Resources/Environment.h"
#include "Materials/Shading/MaterialEvaluation.h"
#include "Materials/SVM/SvmEval.h"

namespace nr::direct_light
{

NR_GPU inline float finiteNonNegative(const float value)
{
    return isfinite(value) && value > 0.0f ? value : 0.0f;
}

// These bounds are uploaded with each candidate. The current flat alias
// sampler uses the scene-independent power proposal; hierarchical samplers
// can use this conservative bound to tighten a position-dependent proposal
// without changing the exact conditional PDFs below.
NR_GPU inline float conservativeSpatialWeight(
    const glm::vec3 position, const DirectLightCandidate& candidate)
{
    const float power = fmaxf(finiteNonNegative(candidate.powerBound),
        finiteNonNegative(candidate.selectionWeight));
    if (power <= 0.0f)
        return 0.0f;
    if (candidate.type == DirectLightType::Directional)
        return power * finiteNonNegative(candidate.orientationBound);

    const glm::vec3 delta = position - candidate.position;
    const float distance = sqrtf(fmaxf(glm::dot(delta, delta), 0.0f));
    const float nearestDistance = fmaxf(distance - candidate.spatialRadius, 0.0f);
    const float orientation = candidate.area > 0.0f
        ? finiteNonNegative(candidate.orientationBound) : 1.0f;
    return power * orientation
        / fmaxf(nearestDistance * nearestDistance, 1.0e-6f);
}

NR_GPU inline float finiteLightMixtureProbability()
{
    return finiteNonNegative(params.scene.finiteLightProbability);
}

NR_GPU inline float environmentLightMixtureProbability()
{
    return finiteNonNegative(params.scene.environmentLightProbability);
}

NR_GPU inline float candidateSelectionPdf(const uint32_t candidateIndex)
{
    if (params.scene.lightAliases != nullptr
        && candidateIndex < params.scene.lightAliasCount)
        return finiteNonNegative(
            params.scene.lightAliases[candidateIndex].selectionPdf);
    if (params.scene.directLightCandidates != nullptr
        && candidateIndex < params.scene.directLightCandidateCount
        && params.scene.lightSelectionWeight > 0.0f)
        return finiteNonNegative(
            params.scene.directLightCandidates[candidateIndex].selectionWeight)
            / finiteNonNegative(params.scene.lightSelectionWeight);
    return 0.0f;
}

// The one canonical finite-light PDF used by direct-light samples and paths
// that hit an analytic or emissive light through BSDF sampling.
NR_GPU inline float lightPdf(
    const uint32_t candidateIndex, const float conditionalPdf)
{
    if (conditionalPdf <= 0.0f)
        return 0.0f;
    return finiteLightMixtureProbability()
        * candidateSelectionPdf(candidateIndex) * conditionalPdf;
}

// The one canonical environment PDF used by direct samples and path misses.
NR_GPU inline float lightPdf(const float environmentConditionalPdf)
{
    if (environmentConditionalPdf <= 0.0f)
        return 0.0f;
    return environmentLightMixtureProbability() * environmentConditionalPdf;
}

NR_GPU inline bool findMeshCandidate(
    const uint32_t instanceIndex, const uint32_t primitiveIndex,
    uint32_t& candidateIndex)
{
    if (params.scene.meshLightCandidateOffsets == nullptr
        || params.scene.meshLightCandidateIndices == nullptr
        || instanceIndex >= params.scene.meshLightInstanceCount)
        return false;
    const uint32_t begin = params.scene.meshLightCandidateOffsets[instanceIndex];
    const uint32_t end = params.scene.meshLightCandidateOffsets[instanceIndex + 1u];
    const uint32_t slot = begin + primitiveIndex;
    if (slot >= end || slot >= params.scene.meshLightCandidateIndexCount)
        return false;
    candidateIndex = params.scene.meshLightCandidateIndices[slot];
    return candidateIndex != InvalidIndex
        && candidateIndex < params.scene.directLightCandidateCount;
}

NR_GPU inline float meshLightConditionalPdf(
    const DirectLightCandidate& candidate,
    const glm::vec3 origin, const glm::vec3 hitPosition,
    const glm::vec3 normal)
{
    const glm::vec3 delta = hitPosition - origin;
    const float distanceSquared = glm::dot(delta, delta);
    if (candidate.area <= 0.0f || distanceSquared <= 1.0e-12f)
        return 0.0f;
    const float cosine = fmaxf(glm::dot(
        normal, -delta / sqrtf(distanceSquared)), 0.0f);
    return cosine > 0.0f
        ? distanceSquared / fmaxf(candidate.area * cosine, 1.0e-20f)
        : 0.0f;
}

NR_GPU inline float meshLightHitPdf(
    const uint32_t instanceIndex, const uint32_t primitiveIndex,
    const glm::vec3 origin, const glm::vec3 hitPosition,
    const glm::vec3 normal)
{
    uint32_t candidateIndex = InvalidIndex;
    if (!findMeshCandidate(instanceIndex, primitiveIndex, candidateIndex))
        return 0.0f;
    const DirectLightCandidate candidate =
        params.scene.directLightCandidates[candidateIndex];
    return lightPdf(candidateIndex, meshLightConditionalPdf(
        candidate, origin, hitPosition, normal));
}

NR_GPU inline float meshLightHitMisWeight(
    const uint32_t instanceIndex, const uint32_t primitiveIndex,
    const glm::vec3 origin, const glm::vec3 hitPosition,
    const glm::vec3 normal, const float bsdfPdf)
{
    if (bsdfPdf <= 0.0f)
        return 1.0f;
    return powerHeuristic(bsdfPdf, meshLightHitPdf(
        instanceIndex, primitiveIndex, origin, hitPosition, normal));
}

NR_GPU inline float candidateSpotAttenuation(
    const DirectLightCandidate& candidate, const glm::vec3 outgoing);

template <typename Rng>
NR_GPU inline bool sampleAnalyticCandidate(
    const DirectLightCandidate& candidate,
    const glm::vec3 position,
    const SampledWavelengths& wavelengths,
    Rng& rng,
    LightSample& light)
{
    if (candidate.type == DirectLightType::Point
        || candidate.type == DirectLightType::Spot)
    {
        light = sampleSphereLight(position, candidate.position,
            candidate.spatialRadius, rng);
        if (light.distance <= 0.0f)
            return false;
        float attenuation = 1.0f;
        if (candidate.type == DirectLightType::Spot)
            attenuation = candidateSpotAttenuation(candidate, -light.direction);
        const glm::vec3 rgb = candidate.spatialRadius > 0.0f
            ? candidate.color * (candidate.intensity * attenuation
                / (LightPi * candidate.spatialRadius * candidate.spatialRadius)
                / fmaxf(light.pdf, 1.0e-20f))
            : candidate.color * (candidate.intensity * attenuation
                / fmaxf(light.distance * light.distance, 1.0e-6f));
        light.radiance = rgbIlluminantToSpectrum(rgb, wavelengths,
            params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
            params.scene.d65);
        return true;
    }

    if (candidate.type == DirectLightType::Rect)
    {
        light = {};
        glm::vec3 sampledPosition = candidate.position;
        light.pdf = sampleSphericalRectangle(position, sampledPosition,
            candidate.tangent, candidate.width,
            candidate.bitangent, candidate.height,
            glm::vec2(randomFloat(rng), randomFloat(rng)));
        if (light.pdf <= 0.0f)
            return false;
        const glm::vec3 delta = sampledPosition - position;
        light.distance = glm::length(delta);
        if (light.distance <= 0.0f)
            return false;
        light.direction = delta / light.distance;
        float emitterCosine = glm::dot(candidate.normal, -light.direction);
        emitterCosine = candidate.twoSided != 0
            ? fabsf(emitterCosine) : fmaxf(emitterCosine, 0.0f);
        if (emitterCosine <= 0.0f)
            return false;
        float barnDoorMask = 1.0f;
        if (candidate.barnDoorEnabled != 0u)
        {
            const glm::vec3 outgoing = -light.direction;
            const float forward = fmaxf(fabsf(glm::dot(
                candidate.normal, outgoing)), 1.0e-5f);
            const glm::vec3 local = sampledPosition - candidate.position;
            const float projectedU = glm::dot(local, candidate.tangent)
                + candidate.barnDoorLength * glm::dot(
                    outgoing, candidate.tangent) / forward;
            const float projectedV = glm::dot(local, candidate.bitangent)
                + candidate.barnDoorLength * glm::dot(
                    outgoing, candidate.bitangent) / forward;
            if (fabsf(projectedU) > candidate.width * 0.5f
                    + candidate.barnDoorExpansion
                || fabsf(projectedV) > candidate.height * 0.5f
                    + candidate.barnDoorExpansion)
                barnDoorMask = 0.0f;
        }
        light.radiance = rgbIlluminantToSpectrum(
            candidate.color * (candidate.intensity * barnDoorMask
                / fmaxf(light.pdf, 1.0e-20f)), wavelengths,
            params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
            params.scene.d65);
        return true;
    }

    if (candidate.type == DirectLightType::Directional)
    {
        light = {};
        light.direction = candidate.normal;
        light.distance = 1.0e16f;
        light.radiance = rgbIlluminantToSpectrum(
            candidate.color * candidate.intensity, wavelengths,
            params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
            params.scene.d65);
        if (candidate.coneOneMinusCosine <= 0.0f)
            return true;
        const UniformConeSample cone = sampleUniformCone(candidate.normal,
            candidate.coneOneMinusCosine,
            glm::vec2(randomFloat(rng), randomFloat(rng)));
        light.direction = cone.direction;
        light.pdf = cone.pdf;
        light.radiance *= 1.0f / fmaxf(
            candidate.coneProjectedArea * light.pdf, 1.0e-20f);
        return true;
    }
    return false;
}

NR_GPU inline float candidateSpotAttenuation(
    const DirectLightCandidate& candidate, const glm::vec3 outgoing)
{
    const float cone = fminf(fmaxf(
        (glm::dot(candidate.normal, outgoing) - candidate.outerCos)
            * candidate.invConeCosineRange, 0.0f), 1.0f);
    return cone * cone * (3.0f - 2.0f * cone);
}

NR_GPU inline LightHit intersectAnalyticCandidate(
    const DirectLightCandidate& candidate,
    const Ray& ray,
    const float tMin,
    const float tMax,
    const SampledWavelengths& wavelengths)
{
    LightHit hit{};
    if (candidate.type == DirectLightType::Point
        || candidate.type == DirectLightType::Spot)
    {
        if (!intersectSphereLight(ray, tMin, tMax, candidate.position,
                candidate.spatialRadius, hit.distance))
            return {};
        hit.pdf = sphereLightPdf(ray.origin(), candidate.position,
            candidate.spatialRadius);
        float attenuation = 1.0f;
        if (candidate.type == DirectLightType::Spot)
        {
            const glm::vec3 hitPosition = ray.at(hit.distance);
            attenuation = candidateSpotAttenuation(candidate,
                nr::safeNormalize(ray.origin() - hitPosition,
                    candidate.normal));
        }
        hit.radiance = rgbIlluminantToSpectrum(candidate.color * (
            candidate.intensity * attenuation
            / fmaxf(LightPi * candidate.spatialRadius
                * candidate.spatialRadius, 1.0e-20f)), wavelengths,
            params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
            params.scene.d65);
        return hit;
    }
    if (candidate.type != DirectLightType::Rect
        || candidate.width <= 0.0f || candidate.height <= 0.0f)
        return hit;
    const float denominator = glm::dot(candidate.normal, ray.direction());
    if (fabsf(denominator) <= 1.0e-8f)
        return hit;
    hit.distance = glm::dot(candidate.position - ray.origin(), candidate.normal)
        / denominator;
    if (hit.distance < tMin || hit.distance > tMax)
        return {};
    const glm::vec3 local = ray.at(hit.distance) - candidate.position;
    if (fabsf(glm::dot(local, candidate.tangent)) > 0.5f * candidate.width
        || fabsf(glm::dot(local, candidate.bitangent)) > 0.5f * candidate.height)
        return {};
    if (candidate.twoSided == 0
        && glm::dot(candidate.normal, -ray.direction()) <= 0.0f)
        return {};
    hit.pdf = sphericalRectanglePdf(ray.origin(), candidate.position,
        ray.at(hit.distance), candidate.tangent, candidate.width,
        candidate.bitangent, candidate.height);
    hit.radiance = rgbIlluminantToSpectrum(
        candidate.color * candidate.intensity, wavelengths,
        params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
        params.scene.d65);
    return hit;
}

NR_GPU inline LightHit intersectDirectionalCandidate(
    const DirectLightCandidate& candidate,
    const Ray& ray,
    const SampledWavelengths& wavelengths)
{
    LightHit hit{};
    const float cosine = glm::dot(ray.direction(), candidate.normal);
    if (candidate.coneOneMinusCosine <= 0.0f)
    {
        if (cosine < 1.0f - 1.0e-7f)
            return hit;
        hit.radiance = rgbIlluminantToSpectrum(
            candidate.color * candidate.intensity, wavelengths,
            params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
            params.scene.d65);
    }
    else
    {
        if (1.0f - cosine > candidate.coneOneMinusCosine)
            return hit;
        hit.pdf = 1.0f / (2.0f * LightPi * candidate.coneOneMinusCosine);
        hit.radiance = rgbIlluminantToSpectrum(
            candidate.color * (candidate.intensity
                / fmaxf(candidate.coneProjectedArea, 1.0e-20f)), wavelengths,
            params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
            params.scene.d65);
    }
    hit.distance = Ray::InfiniteDistance;
    return hit;
}

template <typename Rng>
NR_GPU inline bool sampleMeshTriangle(
    const uint32_t candidateIndex,
    const DirectLightCandidate& candidate,
    const glm::vec3 position,
    const SampledWavelengths& wavelengths,
    Rng& rng,
    LightSample& light)
{
    const float root = sqrtf(randomFloat(rng));
    const float u = 1.0f - root;
    const float v = randomFloat(rng) * root;
    RayHit hit{};
    hit.u = u;
    hit.v = v;
    hit.instanceIndex = candidate.instanceIndex;
    hit.primitiveIndex = candidate.primitiveIndex;
    const Surface surface = Surface::fromHit(params.scene, hit);
    const glm::vec3 delta = surface.position - position;
    const float distanceSquared = glm::dot(delta, delta);
    if (candidate.area <= 0.0f || distanceSquared <= 1.0e-12f)
        return false;
    const float distance = sqrtf(distanceSquared);
    light.direction = delta / distance;
    light.distance = distance;
    light.pdf = meshLightConditionalPdf(
        candidate, position, surface.position, surface.geometricNormal);
    if (light.pdf <= 0.0f)
        return false;

    MaterialEvaluation evaluation{};
    nr::shading::NoorRayCompositeBsdf bsdf(
        surface.geometricNormal, surface.normal, -light.direction);
    MaterialShadingContext context{};
    context.position = surface.position;
    context.geometricNormal = surface.geometricNormal;
    context.interpolatedNormal = surface.normal;
    context.tangent = surface.tangent;
    context.bitangent = glm::cross(surface.normal, surface.tangent)
        * surface.tangentSign;
    context.uv = surface.uv;
    context.vertexColor = surface.color;
    context.viewDirection = -light.direction;
    context.primitiveId = surface.primitiveIndex;
    const GpuInstance instance = params.scene.instances[surface.instanceIndex];
    context.objectToWorld = instance.objectToWorld;
    context.worldToObject = instance.worldToObject;
    context.normalToWorld = instance.normalToWorld;
    if (!nr::svm::svmEvalNodes(params.scene,
        surface.material->svmBytecodeOffset, surface.material->svmBytecodeLength,
        surface.material->svmTextureOffset, surface.material->svmTextureCount,
        context, wavelengths, false, evaluation, bsdf, true)
        || !evaluation.has(MaterialEvaluationFlags::HasEmission))
        return false;
    SampledSpectrum emission = rgbIlluminantToSpectrum(
        evaluation.emission * evaluation.emissionStrength, wavelengths,
        params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
        params.scene.d65);
    emission *= fminf(fmaxf(evaluation.opacity * surface.color.a, 0.0f), 1.0f);
    if (!emission.isFinite() || emission.maxComponent() <= 0.0f)
        return false;
    light.candidateIndex = candidateIndex;
    light.radiance = emission / fmaxf(light.pdf, 1.0e-20f);
    return true;
}

template <typename Rng>
NR_GPU inline bool sampleCandidate(
    const uint32_t candidateIndex,
    const glm::vec3 position,
    const SampledWavelengths& wavelengths,
    Rng& rng,
    LightSample& light)
{
    if (params.scene.directLightCandidates == nullptr
        || candidateIndex >= params.scene.directLightCandidateCount)
        return false;
    const DirectLightCandidate candidate =
        params.scene.directLightCandidates[candidateIndex];
    bool sampled = false;
    if (candidate.type == DirectLightType::MeshTriangle)
        sampled = sampleMeshTriangle(candidateIndex, candidate, position,
            wavelengths, rng, light);
    else
        sampled = sampleAnalyticCandidate(candidate, position, wavelengths, rng, light);
    if (!sampled)
        return false;
    light.candidateIndex = candidateIndex;
    return light.radiance.isFinite() && light.radiance.maxComponent() > 0.0f;
}

template <typename Rng>
NR_GPU inline bool sampleLight(
    const glm::vec3 position,
    const SampledWavelengths& wavelengths,
    Rng& rng,
    LightSample& light,
    float& analyticPdf,
    float& environmentPdf)
{
    analyticPdf = 0.0f;
    environmentPdf = 0.0f;
    light = {};

    const float finiteProbability = finiteLightMixtureProbability();
    const float environmentProbability = environmentLightMixtureProbability();
    if (finiteProbability <= 0.0f && environmentProbability <= 0.0f)
        return false;

    if (finiteProbability > 0.0f
        && randomFloat(rng) < finiteProbability)
    {
        uint32_t candidateIndex = InvalidIndex;
        if (params.scene.lightAliases != nullptr
            && params.scene.lightAliasCount > 0)
        {
            const uint32_t count = params.scene.lightAliasCount;
            const float sample = randomFloat(rng) * static_cast<float>(count);
            const uint32_t slot = min(static_cast<uint32_t>(sample), count - 1u);
            const float fraction = sample - static_cast<float>(slot);
            const LightAliasEntry entry = params.scene.lightAliases[slot];
            candidateIndex = fraction < entry.threshold ? slot : entry.alias;
        }
        else if (params.scene.directLightCandidates != nullptr)
        {
            // Defensive fallback for a transient scene-upload failure. The
            // normal path is O(1) alias sampling; this preserves the exact
            // proposal distribution if only the alias buffer is unavailable.
            const float target = randomFloat(rng)
                * finiteNonNegative(params.scene.lightSelectionWeight);
            float cumulative = 0.0f;
            for (uint32_t i = 0; i < params.scene.directLightCandidateCount; ++i)
            {
                cumulative += finiteNonNegative(
                    params.scene.directLightCandidates[i].selectionWeight);
                if (target <= cumulative)
                {
            candidateIndex = i;
                    break;
                }
            }
        }
        if (candidateIndex == InvalidIndex
            || !sampleCandidate(candidateIndex, position, wavelengths, rng, light))
            return false;

        const float selectionPdf = lightPdf(candidateIndex, 1.0f);
        if (selectionPdf <= 0.0f)
            return false;
        if (light.pdf > 0.0f)
            analyticPdf = selectionPdf * light.pdf;
        light.radiance *= 1.0f / selectionPdf;
        return light.radiance.isFinite() && light.radiance.maxComponent() > 0.0f;
    }

    if (environmentProbability <= 0.0f)
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
    environmentPdf = lightPdf(sample.pdf);
    return light.radiance.isFinite() && light.radiance.maxComponent() > 0.0f;
}

// Compatibility name for the two shading kernels. All light sampling now
// flows through sampleLight(), so there is only one selection/PDF policy.
template <typename Rng>
NR_GPU inline bool sampleDirectLight(
    const glm::vec3 position,
    const SampledWavelengths& wavelengths,
    Rng& rng,
    LightSample& light,
    float& analyticPdf,
    float& environmentPdf)
{
    return sampleLight(position, wavelengths, rng, light,
        analyticPdf, environmentPdf);
}

} // namespace nr::direct_light
