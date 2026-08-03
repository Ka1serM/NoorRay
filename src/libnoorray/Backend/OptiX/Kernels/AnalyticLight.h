#pragma once

#include <optix_device.h>

#include "Backend/OptiX/ABI/SceneData.h"
#include "Rendering/MisHeuristic.h"
#include "Rendering/Lighting/DirectLightSampling.h"

extern "C"
{
extern __constant__ KernelParams params;
}

namespace nr::light_bvh
{

NR_GPU inline bool intersectMeshTriangle(
    const DirectLightCandidate& candidate,
    const glm::vec3 origin,
    const glm::vec3 direction,
    const float tMin,
    const float tMax,
    float& distance,
    float& barycentricU,
    float& barycentricV)
{
    if (candidate.instanceIndex == InvalidIndex
        || candidate.primitiveIndex == InvalidIndex
        || candidate.instanceIndex >= params.scene.meshInstanceCount)
        return false;

    const GpuInstance instance = params.scene.instances[candidate.instanceIndex];
    const MeshAsset& mesh = params.scene.meshes[instance.meshIndex];
    const auto& indices = mesh.getIndices();
    const auto& vertices = mesh.getVertices();
    const size_t indexOffset = static_cast<size_t>(candidate.primitiveIndex) * 3u;
    if (indexOffset + 2u >= indices.size())
        return false;
    const uint32_t i0 = indices[indexOffset];
    const uint32_t i1 = indices[indexOffset + 1u];
    const uint32_t i2 = indices[indexOffset + 2u];
    if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size())
        return false;

    const glm::vec3 a = glm::vec3(
        instance.objectToWorld * glm::vec4(vertices[i0].position, 1.0f));
    const glm::vec3 b = glm::vec3(
        instance.objectToWorld * glm::vec4(vertices[i1].position, 1.0f));
    const glm::vec3 c = glm::vec3(
        instance.objectToWorld * glm::vec4(vertices[i2].position, 1.0f));
    const glm::vec3 edge1 = b - a;
    const glm::vec3 edge2 = c - a;
    const glm::vec3 p = glm::cross(direction, edge2);
    const float determinant = glm::dot(edge1, p);
    if (fabsf(determinant) <= 1.0e-8f)
        return false;

    const float inverseDeterminant = 1.0f / determinant;
    const glm::vec3 t = origin - a;
    barycentricU = glm::dot(t, p) * inverseDeterminant;
    if (barycentricU < 0.0f || barycentricU > 1.0f)
        return false;
    const glm::vec3 q = glm::cross(t, edge1);
    barycentricV = glm::dot(direction, q) * inverseDeterminant;
    if (barycentricV < 0.0f
        || barycentricU + barycentricV > 1.0f)
        return false;
    distance = glm::dot(edge2, q) * inverseDeterminant;
    return distance >= tMin && distance <= tMax;
}

NR_GPU inline bool intersectAnalyticShape(
    const DirectLightCandidate& candidate,
    const glm::vec3 origin,
    const glm::vec3 direction,
    const float tMin,
    const float tMax,
    float& distance)
{
    if (candidate.type == DirectLightType::Point
        || candidate.type == DirectLightType::Spot)
    {
        return intersectSphereLight(
            Ray(origin, direction), tMin, tMax,
            candidate.position, candidate.spatialRadius, distance);
    }

    if (candidate.type != DirectLightType::Rect
        || candidate.width <= 0.0f || candidate.height <= 0.0f)
        return false;

    const glm::vec3 normal = nr::safeNormalize(candidate.normal);
    const glm::vec3 tangent = nr::safeNormalize(candidate.tangent);
    const glm::vec3 bitangent = nr::safeNormalize(glm::cross(normal, tangent));
    const float denominator = glm::dot(normal, direction);
    if (glm::dot(normal, normal) <= 0.0f
        || glm::dot(tangent, tangent) <= 0.0f
        || glm::dot(bitangent, bitangent) <= 0.0f
        || fabsf(denominator) <= 1.0e-8f)
        return false;

    distance = glm::dot(candidate.position - origin, normal) / denominator;
    if (distance < tMin || distance > tMax)
        return false;
    const glm::vec3 local = origin + direction * distance - candidate.position;
    if (fabsf(glm::dot(local, tangent)) > 0.5f * candidate.width
        || fabsf(glm::dot(local, bitangent)) > 0.5f * candidate.height)
        return false;
    return candidate.twoSided != 0 || glm::dot(normal, -direction) > 0.0f;
}

NR_GPU inline LightHit intersectAnalyticCandidate(
    const uint32_t candidateIndex,
    const Ray& ray,
    const float tMin,
    const float tMax,
    const SampledWavelengths& wavelengths)
{
    const DirectLightCandidate& candidate =
        params.scene.directLightCandidates[candidateIndex];
    return nr::direct_light::intersectAnalyticCandidate(
        candidate, ray, tMin, tMax, wavelengths);
}

NR_GPU inline bool evaluateMeshEmission(
    const uint32_t candidateIndex,
    const Ray& ray,
    const float barycentricU,
    const float barycentricV,
    const SampledWavelengths& wavelengths,
    SampledSpectrum& emission,
    float& conditionalPdf)
{
    const DirectLightCandidate& candidate =
        params.scene.directLightCandidates[candidateIndex];
    RayHit hit{};
    hit.u = barycentricU;
    hit.v = barycentricV;
    hit.instanceIndex = candidate.instanceIndex;
    hit.primitiveIndex = candidate.primitiveIndex;
    const Surface surface = Surface::fromHit(params.scene, hit);
    conditionalPdf = nr::direct_light::meshLightConditionalPdf(
        candidate, ray.origin(), surface.position, surface.geometricNormal);

    MaterialEvaluation evaluation{};
    nr::shading::NoorRayCompositeBsdf bsdf(
        surface.geometricNormal, surface.normal, -ray.direction());
    MaterialShadingContext context{};
    context.position = surface.position;
    context.geometricNormal = surface.geometricNormal;
    context.interpolatedNormal = surface.normal;
    context.tangent = surface.tangent;
    context.bitangent = glm::cross(surface.normal, surface.tangent)
        * surface.tangentSign;
    context.uv = surface.uv;
    context.vertexColor = surface.color;
    context.viewDirection = -ray.direction();
    context.primitiveId = surface.primitiveIndex;
    const GpuInstance instance = params.scene.instances[surface.instanceIndex];
    context.objectToWorld = instance.objectToWorld;
    context.worldToObject = instance.worldToObject;
    context.normalToWorld = instance.normalToWorld;
    if (!nr::svm::svmEvalNodes(
        params.scene, surface.material->svmBytecodeOffset,
        surface.material->svmBytecodeLength, surface.material->svmTextureOffset,
        surface.material->svmTextureCount, context, wavelengths, false,
        evaluation, bsdf, true)
        || !evaluation.has(MaterialEvaluationFlags::HasEmission))
        return false;

    emission = rgbIlluminantToSpectrum(
        evaluation.emission * evaluation.emissionStrength, wavelengths,
        params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
        params.scene.d65);
    emission *= fminf(fmaxf(evaluation.opacity * surface.color.a, 0.0f), 1.0f);
    return emission.isFinite() && emission.maxComponent() > 0.0f;
}

NR_GPU inline void handleClosestHit(PathTracePayload& payload)
{
    PathState& state = *payload.state;
    const uint32_t primitive = optixGetPrimitiveIndex();
    const bool meshLightBvh = optixGetInstanceId() == 1u;
    const uint32_t* const candidateIndices = meshLightBvh
        ? params.scene.meshLightBvhCandidateIndices
        : params.scene.analyticLightBvhCandidateIndices;
    const uint32_t primitiveCount = meshLightBvh
        ? params.scene.meshLightBvhPrimitiveCount
        : params.scene.analyticLightBvhPrimitiveCount;
    if (candidateIndices == nullptr || primitive >= primitiveCount
        || params.scene.directLightCandidates == nullptr)
    {
        payload.status = PathTraceStatus::Terminate;
        return;
    }

    const uint32_t candidateIndex =
        candidateIndices[primitive];
    if (candidateIndex >= params.scene.directLightCandidateCount)
    {
        payload.status = PathTraceStatus::Terminate;
        return;
    }
    const DirectLightCandidate& candidate =
        params.scene.directLightCandidates[candidateIndex];
    const Ray ray(
        glm::vec3(optixGetWorldRayOrigin().x, optixGetWorldRayOrigin().y,
            optixGetWorldRayOrigin().z),
        glm::vec3(optixGetWorldRayDirection().x, optixGetWorldRayDirection().y,
            optixGetWorldRayDirection().z));

    SampledSpectrum contribution(0.0f);
    float conditionalPdf = 0.0f;
    if (candidate.type == DirectLightType::MeshTriangle)
    {
        SampledSpectrum emission(0.0f);
        const float u = __uint_as_float(optixGetAttribute_0());
        const float v = __uint_as_float(optixGetAttribute_1());
        if (evaluateMeshEmission(candidateIndex, ray, u, v, state.wl,
                emission, conditionalPdf))
            contribution = emission;
    }
    else
    {
        const LightHit light = intersectAnalyticCandidate(
            candidateIndex, ray, optixGetRayTmin(), optixGetRayTmax(), state.wl);
        contribution = light.radiance;
        conditionalPdf = light.pdf;
    }

    if (contribution.maxComponent() <= 0.0f || !contribution.isFinite())
    {
        payload.status = PathTraceStatus::Terminate;
        return;
    }
    float misWeight = 1.0f;
    if (state.lastBsdfPdf > 0.0f && conditionalPdf > 0.0f)
        misWeight = powerHeuristic(state.lastBsdfPdf,
            nr::direct_light::lightPdf(candidateIndex, conditionalPdf));
    state.radiance += state.throughput * contribution * misWeight;
    payload.status = PathTraceStatus::Terminate;
}

}

extern "C" __global__ void __intersection__analyticLight()
{
    const uint32_t primitive = optixGetPrimitiveIndex();
    const bool meshLightBvh = optixGetInstanceId() == 1u;
    const uint32_t* const candidateIndices = meshLightBvh
        ? params.scene.meshLightBvhCandidateIndices
        : params.scene.analyticLightBvhCandidateIndices;
    const uint32_t primitiveCount = meshLightBvh
        ? params.scene.meshLightBvhPrimitiveCount
        : params.scene.analyticLightBvhPrimitiveCount;
    if (candidateIndices == nullptr || primitive >= primitiveCount
        || params.scene.directLightCandidates == nullptr)
        return;
    const uint32_t candidateIndex =
        candidateIndices[primitive];
    if (candidateIndex >= params.scene.directLightCandidateCount)
        return;
    const DirectLightCandidate& candidate =
        params.scene.directLightCandidates[candidateIndex];
    const glm::vec3 origin(
        optixGetObjectRayOrigin().x, optixGetObjectRayOrigin().y,
        optixGetObjectRayOrigin().z);
    const glm::vec3 direction(
        optixGetObjectRayDirection().x, optixGetObjectRayDirection().y,
        optixGetObjectRayDirection().z);
    float distance = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    bool hit = false;
    if (candidate.type == DirectLightType::MeshTriangle)
        hit = nr::light_bvh::intersectMeshTriangle(
            candidate, origin, direction, optixGetRayTmin(),
            optixGetRayTmax(), distance, u, v);
    else
        hit = nr::light_bvh::intersectAnalyticShape(
            candidate, origin, direction, optixGetRayTmin(),
            optixGetRayTmax(), distance);
    if (hit)
    {
        if (candidate.type == DirectLightType::MeshTriangle)
            optixReportIntersection(distance, 0,
                __float_as_uint(u), __float_as_uint(v));
        else
            optixReportIntersection(distance, 0);
    }
}

extern "C" __global__ void __closesthit__analyticLight()
{
    PathTracePayload* const payload = getPathTracePayload<>();
    nr::light_bvh::handleClosestHit(*payload);
}
