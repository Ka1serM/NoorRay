#pragma once

#include <cmath>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "Backend/Host/Platform.h"
#include "Rendering/Sampling/HemisphereSampler.h"
#include "Rendering/Sampling/RandomSampler.h"
#include "Materials/Shading/BsdfClosure.h"
#include "Materials/Shading/Dielectric.h"
#include "Materials/Shading/EnergyLut.h"

namespace nr::shading::ggx
{

inline constexpr float Pi = BsdfPi;
inline constexpr float AlmostSpecularAlphaProduct = 2.0e-10f;
inline constexpr float SingularPdf = BsdfSingularPdf;
inline constexpr float MinimumLutRoughness = 0.035f;
inline constexpr float DenominatorEpsilon = 1.0e-6f;

NR_CPU_GPU inline float alphaFromPerceptualRoughness(const float roughness)
{
    return roughness * roughness;
}

NR_CPU_GPU inline bool isAlmostSpecular(const float roughness)
{
    const float alpha = alphaFromPerceptualRoughness(roughness);
    return alpha * alpha <= AlmostSpecularAlphaProduct;
}

NR_CPU_GPU inline float distribution(
    const float normalHalfCosine, const float roughness)
{
    if (isAlmostSpecular(roughness))
        return 0.0f;
    const float alpha = alphaFromPerceptualRoughness(roughness);
    const float alphaSquared = alpha * alpha;
    const float cosine = fmaxf(normalHalfCosine, 0.0f);
    const float cosineSquared = fminf(cosine * cosine, 1.0f);
    const float denominator = (1.0f - cosineSquared)
        + alphaSquared * cosineSquared;
    return alphaSquared / (Pi * denominator * denominator);
}

NR_CPU_GPU inline float distribution(
    const glm::vec3 normal,
    const glm::vec3 halfVector,
    const float roughness)
{
    return distribution(glm::dot(normal, halfVector), roughness);
}

NR_CPU_GPU inline float smithG1(
    const float normalDirectionCosine, const float roughness)
{
    const float cosine = fminf(fmaxf(normalDirectionCosine, 0.0f), 1.0f);
    if (cosine <= 0.0f)
        return 0.0f;
    const float alpha = alphaFromPerceptualRoughness(roughness);
    const float alphaSquared = alpha * alpha;
    const float root = sqrtf(alphaSquared
        + (1.0f - alphaSquared) * cosine * cosine);
    return 2.0f * cosine / (cosine + root);
}

NR_CPU_GPU inline float smithLambda(
    const float normalDirectionCosine, const float roughness)
{
    const float cosine = fminf(fmaxf(normalDirectionCosine, 0.0f), 1.0f);
    if (cosine <= 0.0f)
        return 1.0e20f;
    const float alpha = alphaFromPerceptualRoughness(roughness);
    const float alphaSquared = alpha * alpha;
    const float tangentSquared = fmaxf(
        1.0f / (cosine * cosine) - 1.0f, 0.0f);
    return 0.5f * (sqrtf(1.0f + alphaSquared * tangentSquared) - 1.0f);
}

NR_CPU_GPU inline float smithG2(
    const float normalViewCosine,
    const float normalLightCosine,
    const float roughness)
{
    if (normalViewCosine <= 0.0f || normalLightCosine <= 0.0f)
        return 0.0f;
    return 1.0f / (1.0f
        + smithLambda(normalViewCosine, roughness)
        + smithLambda(normalLightCosine, roughness));
}

NR_CPU_GPU inline float smithG2(
    const glm::vec3 normal,
    const glm::vec3 view,
    const glm::vec3 light,
    const float roughness)
{
    return smithG2(glm::dot(normal, view),
        glm::dot(normal, light), roughness);
}

NR_CPU_GPU inline float reflectionPdf(
    const glm::vec3 normal,
    const glm::vec3 view,
    const glm::vec3 halfVector,
    const float roughness)
{
    const float normalView = glm::dot(normal, view);
    if (normalView <= 0.0f || isAlmostSpecular(roughness))
        return 0.0f;
    return distribution(normal, halfVector, roughness)
        * smithG1(normalView, roughness) / (4.0f * normalView);
}

NR_CPU_GPU inline glm::vec3 sampleVisibleNormalLocal(
    const glm::vec3 view,
    const float roughness,
    const glm::vec2 sample)
{
    const float alpha = alphaFromPerceptualRoughness(roughness);
    const glm::vec3 stretchedView = nr::safeNormalize(
        glm::vec3(alpha * view.x, alpha * view.y, view.z),
        glm::vec3(0.0f, 0.0f, 1.0f));

    const float lengthSquared = stretchedView.x * stretchedView.x
        + stretchedView.y * stretchedView.y;
    const glm::vec3 tangent1 = lengthSquared > 1.0e-7f
        ? glm::vec3(-stretchedView.y, stretchedView.x, 0.0f)
            / sqrtf(lengthSquared)
        : glm::vec3(1.0f, 0.0f, 0.0f);
    const glm::vec3 tangent2 = lengthSquared > 1.0e-7f
        ? glm::cross(stretchedView, tangent1)
        : glm::vec3(0.0f, 1.0f, 0.0f);

    glm::vec2 disk = nr::sampling::concentricDisk(sample);
    disk.y = sqrtf(fmaxf(1.0f - disk.x * disk.x, 0.0f))
            * (1.0f - 0.5f * (1.0f + stretchedView.z))
        + disk.y * (0.5f * (1.0f + stretchedView.z));

    const float hemisphereZ = sqrtf(fmaxf(
        1.0f - disk.x * disk.x - disk.y * disk.y, 0.0f));
    const glm::vec3 halfStretched = tangent1 * disk.x
        + tangent2 * disk.y + stretchedView * hemisphereZ;
    return nr::safeNormalize(glm::vec3(
        alpha * halfStretched.x,
        alpha * halfStretched.y,
        fmaxf(halfStretched.z, 0.0f)),
        glm::vec3(0.0f, 0.0f, 1.0f));
}

NR_CPU_GPU inline glm::vec3 sampleVisibleNormal(
    const glm::vec3 normal,
    const glm::vec3 view,
    const float roughness,
    const glm::vec2 sample)
{
    glm::vec3 tangent{}, bitangent{};
    nr::sampling::buildBasis(normal, tangent, bitangent);
    const glm::vec3 localView(
        glm::dot(view, tangent),
        glm::dot(view, bitangent),
        glm::dot(view, normal));
    const glm::vec3 localHalf = sampleVisibleNormalLocal(
        localView, roughness, sample);
    return nr::safeNormalize(
        tangent * localHalf.x + bitangent * localHalf.y
        + normal * localHalf.z, normal);
}

// Blend dielectric and conductor Fresnel at the same half-vector cosine.
NR_CPU_GPU inline float fresnelBlend(
    const float dielectricFresnel,
    const float viewHalfOrNormalCosine,
    const float albedo,
    const float metallic)
{
    const float conductorFresnel =
        nr::shading::dielectric::fresnelFromNormalReflectance(
            viewHalfOrNormalCosine, albedo);
    return dielectricFresnel
        + (conductorFresnel - dielectricFresnel) * metallic;
}

// Shared GGX reflection/transmission geometry. `cosineWeighted` is f*N.L;
// Fresnel remains outside so closures can evaluate spectral Fresnel once per
// wavelength while sharing the half-vector and PDF.
struct Evaluation
{
    glm::vec3 halfVector{};
    float viewHalfCosine{};
    float cosineWeighted{};
    float pdf{};
};

NR_CPU_GPU inline Evaluation evaluateReflection(
    const glm::vec3 normal,
    const glm::vec3 view,
    const glm::vec3 outgoing,
    const float roughness)
{
    Evaluation result{};
    const float normalView = glm::dot(normal, view);
    const float normalOutgoing = glm::dot(normal, outgoing);
    if (normalView <= 0.0f || normalOutgoing <= 0.0f
        || isAlmostSpecular(roughness))
        return result;

    const glm::vec3 sum = view + outgoing;
    const float sumLengthSquared = glm::dot(sum, sum);
    if (sumLengthSquared <= 0.0f)
        return result;
    result.halfVector = sum / sqrtf(sumLengthSquared);
    result.viewHalfCosine = glm::dot(result.halfVector, view);
    if (result.viewHalfCosine <= 0.0f)
        return {};

    const float distributionValue = distribution(
        normal, result.halfVector, roughness);
    const float masking = smithG1(normalView, roughness);
    const float shadowingMasking = smithG2(
        normalView, normalOutgoing, roughness);
    const float common = 0.25f * distributionValue / normalView;
    result.pdf = common * masking;
    result.cosineWeighted = common * shadowingMasking;
    return result;
}

NR_CPU_GPU inline Evaluation evaluateTransmission(
    const glm::vec3 normal,
    const glm::vec3 view,
    const glm::vec3 outgoing,
    const float roughness,
    const float etaTransmittedOverIncident)
{
    Evaluation result{};
    const float normalView = glm::dot(normal, view);
    const float normalOutgoing = glm::dot(normal, outgoing);
    if (normalView <= 0.0f || normalOutgoing >= 0.0f
        || isAlmostSpecular(roughness)
        || fabsf(etaTransmittedOverIncident - 1.0f) < 1.0e-4f)
        return result;

    const glm::vec3 unnormalizedHalf =
        -(etaTransmittedOverIncident * outgoing + view);
    const float halfLengthSquared = glm::dot(
        unnormalizedHalf, unnormalizedHalf);
    if (halfLengthSquared <= 0.0f)
        return result;
    const float inverseHalfLength = 1.0f / sqrtf(halfLengthSquared);
    result.halfVector = unnormalizedHalf * inverseHalfLength;
    if (glm::dot(normal, result.halfVector) < 0.0f)
        result.halfVector = -result.halfVector;

    result.viewHalfCosine = glm::dot(result.halfVector, view);
    const float outgoingHalfCosine = glm::dot(
        result.halfVector, outgoing);
    if (result.viewHalfCosine <= 0.0f || outgoingHalfCosine >= 0.0f)
        return {};

    const float distributionValue = distribution(
        normal, result.halfVector, roughness);
    const float masking = smithG1(normalView, roughness);
    const float shadowingMasking = smithG2(
        normalView, -normalOutgoing, roughness);
    const float jacobian = etaTransmittedOverIncident
        * etaTransmittedOverIncident * inverseHalfLength * inverseHalfLength
        * fabsf(result.viewHalfCosine * outgoingHalfCosine);
    const float common = distributionValue * jacobian / normalView;
    result.pdf = common * masking;
    result.cosineWeighted = common * shadowingMasking;
    return result;
}

}
