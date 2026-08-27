#pragma once

// A 1:1 port of Materials/Shading/Ggx.h.

#include "BsdfClosure.slang"
#include "Dielectric.slang"

static const float GgxAlmostSpecularAlphaProduct = 2.0e-10;
static const float GgxMinimumLutRoughness = 0.035;
static const float GgxDenominatorEpsilon = 1.0e-6;

float3 safeNormalize(float3 value, float3 fallback)
{
    const float lengthSquared = dot(value, value);
    return lengthSquared > 0.0 ? value * rsqrt(lengthSquared) : fallback;
}

float alphaFromPerceptualRoughness(float roughness)
{
    return roughness * roughness;
}

bool isAlmostSpecular(float roughness)
{
    const float alpha = alphaFromPerceptualRoughness(roughness);
    return alpha * alpha <= GgxAlmostSpecularAlphaProduct;
}

float ggxDistribution(float normalHalfCosine, float roughness)
{
    if (isAlmostSpecular(roughness))
        return 0.0;
    const float alpha = alphaFromPerceptualRoughness(roughness);
    const float alphaSquared = alpha * alpha;
    const float cosine = max(normalHalfCosine, 0.0);
    const float cosineSquared = min(cosine * cosine, 1.0);
    const float denominator = (1.0 - cosineSquared) + alphaSquared * cosineSquared;
    return alphaSquared / (BsdfPi * denominator * denominator);
}

float ggxDistribution(float3 normal, float3 halfVector, float roughness)
{
    return ggxDistribution(dot(normal, halfVector), roughness);
}

float smithG1(float normalDirectionCosine, float roughness)
{
    const float cosine = clamp(normalDirectionCosine, 0.0, 1.0);
    if (cosine <= 0.0)
        return 0.0;
    const float alpha = alphaFromPerceptualRoughness(roughness);
    const float alphaSquared = alpha * alpha;
    const float root = sqrt(alphaSquared + (1.0 - alphaSquared) * cosine * cosine);
    return 2.0 * cosine / (cosine + root);
}

float smithLambda(float normalDirectionCosine, float roughness)
{
    const float cosine = clamp(normalDirectionCosine, 0.0, 1.0);
    if (cosine <= 0.0)
        return 1.0e20;
    const float alpha = alphaFromPerceptualRoughness(roughness);
    const float alphaSquared = alpha * alpha;
    const float tangentSquared = max(1.0 / (cosine * cosine) - 1.0, 0.0);
    return 0.5 * (sqrt(1.0 + alphaSquared * tangentSquared) - 1.0);
}

float smithG2(float normalViewCosine, float normalLightCosine, float roughness)
{
    if (normalViewCosine <= 0.0 || normalLightCosine <= 0.0)
        return 0.0;
    return 1.0 / (1.0 + smithLambda(normalViewCosine, roughness)
        + smithLambda(normalLightCosine, roughness));
}

float smithG2(float3 normal, float3 view, float3 light, float roughness)
{
    return smithG2(dot(normal, view), dot(normal, light), roughness);
}

float ggxReflectionPdf(float3 normal, float3 view, float3 halfVector, float roughness)
{
    const float normalView = dot(normal, view);
    if (normalView <= 0.0 || isAlmostSpecular(roughness))
        return 0.0;
    return ggxDistribution(normal, halfVector, roughness)
        * smithG1(normalView, roughness) / (4.0 * normalView);
}

float3 sampleVisibleNormalLocal(float3 view, float roughness, float2 sample)
{
    const float alpha = alphaFromPerceptualRoughness(roughness);
    const float3 stretchedView = safeNormalize(
        float3(alpha * view.x, alpha * view.y, view.z), float3(0.0, 0.0, 1.0));

    const float lengthSquared = stretchedView.x * stretchedView.x
        + stretchedView.y * stretchedView.y;
    const float3 tangent1 = lengthSquared > 1.0e-7
        ? float3(-stretchedView.y, stretchedView.x, 0.0) / sqrt(lengthSquared)
        : float3(1.0, 0.0, 0.0);
    const float3 tangent2 = lengthSquared > 1.0e-7
        ? cross(stretchedView, tangent1)
        : float3(0.0, 1.0, 0.0);

    float2 disk = concentricDisk(sample);
    disk.y = sqrt(max(1.0 - disk.x * disk.x, 0.0))
            * (1.0 - 0.5 * (1.0 + stretchedView.z))
        + disk.y * (0.5 * (1.0 + stretchedView.z));

    const float hemisphereZ = sqrt(max(1.0 - disk.x * disk.x - disk.y * disk.y, 0.0));
    const float3 halfStretched = tangent1 * disk.x + tangent2 * disk.y
        + stretchedView * hemisphereZ;
    return safeNormalize(float3(
        alpha * halfStretched.x,
        alpha * halfStretched.y,
        max(halfStretched.z, 0.0)), float3(0.0, 0.0, 1.0));
}

float3 sampleVisibleNormal(float3 normal, float3 view, float roughness, float2 sample)
{
    float3 tangent, bitangent;
    buildBasis(normal, tangent, bitangent);
    const float3 localView = float3(
        dot(view, tangent), dot(view, bitangent), dot(view, normal));
    const float3 localHalf = sampleVisibleNormalLocal(localView, roughness, sample);
    return safeNormalize(
        tangent * localHalf.x + bitangent * localHalf.y + normal * localHalf.z, normal);
}

// Blend dielectric and conductor Fresnel at the same half-vector cosine.
float fresnelBlend(float dielectricFresnelValue, float viewHalfOrNormalCosine,
    float albedo, float metallic)
{
    const float conductor = fresnelFromNormalReflectance(viewHalfOrNormalCosine, albedo);
    return dielectricFresnelValue + (conductor - dielectricFresnelValue) * metallic;
}

// Shared GGX reflection/transmission geometry. `cosineWeighted` is f*N.L;
// Fresnel stays outside so closures can evaluate spectral Fresnel once per
// wavelength while sharing the half-vector and PDF.
struct GgxEvaluation
{
    float3 halfVector;
    float viewHalfCosine;
    float cosineWeighted;
    float pdf;
};

GgxEvaluation makeGgxEvaluation()
{
    GgxEvaluation result;
    result.halfVector = float3(0.0);
    result.viewHalfCosine = 0.0;
    result.cosineWeighted = 0.0;
    result.pdf = 0.0;
    return result;
}

GgxEvaluation evaluateGgxReflection(float3 normal, float3 view, float3 outgoing,
    float roughness)
{
    GgxEvaluation result = makeGgxEvaluation();
    const float normalView = dot(normal, view);
    const float normalOutgoing = dot(normal, outgoing);
    if (normalView <= 0.0 || normalOutgoing <= 0.0 || isAlmostSpecular(roughness))
        return result;

    const float3 sum = view + outgoing;
    const float sumLengthSquared = dot(sum, sum);
    if (sumLengthSquared <= 0.0)
        return result;
    result.halfVector = sum / sqrt(sumLengthSquared);
    result.viewHalfCosine = dot(result.halfVector, view);
    if (result.viewHalfCosine <= 0.0)
        return makeGgxEvaluation();

    const float distributionValue = ggxDistribution(normal, result.halfVector, roughness);
    const float masking = smithG1(normalView, roughness);
    const float shadowingMasking = smithG2(normalView, normalOutgoing, roughness);
    const float common = 0.25 * distributionValue / normalView;
    result.pdf = common * masking;
    result.cosineWeighted = common * shadowingMasking;
    return result;
}

GgxEvaluation evaluateGgxTransmission(float3 normal, float3 view, float3 outgoing,
    float roughness, float etaTransmittedOverIncident)
{
    GgxEvaluation result = makeGgxEvaluation();
    const float normalView = dot(normal, view);
    const float normalOutgoing = dot(normal, outgoing);
    if (normalView <= 0.0 || normalOutgoing >= 0.0 || isAlmostSpecular(roughness)
        || abs(etaTransmittedOverIncident - 1.0) < 1.0e-4)
        return result;

    const float3 unnormalizedHalf = -(etaTransmittedOverIncident * outgoing + view);
    const float halfLengthSquared = dot(unnormalizedHalf, unnormalizedHalf);
    if (halfLengthSquared <= 0.0)
        return result;
    const float inverseHalfLength = 1.0 / sqrt(halfLengthSquared);
    result.halfVector = unnormalizedHalf * inverseHalfLength;
    if (dot(normal, result.halfVector) < 0.0)
        result.halfVector = -result.halfVector;

    result.viewHalfCosine = dot(result.halfVector, view);
    const float outgoingHalfCosine = dot(result.halfVector, outgoing);
    if (result.viewHalfCosine <= 0.0 || outgoingHalfCosine >= 0.0)
        return makeGgxEvaluation();

    const float distributionValue = ggxDistribution(normal, result.halfVector, roughness);
    const float masking = smithG1(normalView, roughness);
    const float shadowingMasking = smithG2(normalView, -normalOutgoing, roughness);
    const float jacobian = etaTransmittedOverIncident * etaTransmittedOverIncident
        * inverseHalfLength * inverseHalfLength
        * abs(result.viewHalfCosine * outgoingHalfCosine);
    const float common = distributionValue * jacobian / normalView;
    result.pdf = common * masking;
    result.cosineWeighted = common * shadowingMasking;
    return result;
}
