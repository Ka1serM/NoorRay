#pragma once

#include <cmath>
#include <cstdint>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "GPU/Annotations.h"
#include "Kernels/Samplers.h"

static constexpr float BsdfPi = 3.14159265358979323846f;
static constexpr float BsdfEpsilon = 1e-5f;

enum class BsdfEvent : uint32_t
{
    Diffuse,
    Specular,
    Transmission,
};

struct BsdfSample
{
    glm::vec3 direction{};
    glm::vec3 weight{};
    BsdfEvent event{BsdfEvent::Diffuse};
};

NR_CPU_GPU inline void buildBsdfBasis(
    const glm::vec3 normal, glm::vec3& tangent, glm::vec3& bitangent)
{
    const glm::vec3 helper = fabsf(normal.z) < 0.999f
        ? glm::vec3(0.0f, 0.0f, 1.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);
    tangent = glm::normalize(glm::cross(normal, helper));
    bitangent = glm::cross(tangent, normal);
}

NR_CPU_GPU inline glm::vec3 sampleDiffuseDirection(const glm::vec3 normal, uint32_t& rng)
{
    const float u1 = randomFloat(rng);
    const float u2 = randomFloat(rng);
    const float radius = sqrtf(u1);
    const float phi = 2.0f * BsdfPi * u2;
    glm::vec3 tangent{}, bitangent{};
    buildBsdfBasis(normal, tangent, bitangent);
    return glm::normalize(
        tangent * (radius * cosf(phi))
        + bitangent * (radius * sinf(phi))
        + normal * sqrtf(fmaxf(0.0f, 1.0f - u1)));
}

NR_CPU_GPU inline glm::vec3 sampleGgxVndfLocal(
    const glm::vec3 view, const float roughness, const glm::vec2 sample)
{
    const float alpha = roughness * roughness;
    const glm::vec3 stretched = glm::normalize(glm::vec3(alpha * view.x, alpha * view.y, view.z));
    const float phi = 2.0f * BsdfPi * sample.x;
    const float z = (1.0f - sample.y) * (1.0f + stretched.z) - stretched.z;
    const float sinTheta = sqrtf(fminf(fmaxf(1.0f - z * z, 0.0f), 1.0f));
    const glm::vec3 c(sinTheta * cosf(phi), sinTheta * sinf(phi), z);
    const glm::vec3 halfStretched = c + stretched;
    return glm::normalize(glm::vec3(
        alpha * halfStretched.x, alpha * halfStretched.y, halfStretched.z));
}

NR_CPU_GPU inline glm::vec3 sampleGgxHalfVector(
    const glm::vec3 view, const glm::vec3 normal, const float roughness, uint32_t& rng)
{
    glm::vec3 tangent{}, bitangent{};
    buildBsdfBasis(normal, tangent, bitangent);
    const glm::vec3 localView(
        glm::dot(view, tangent), glm::dot(view, bitangent), glm::dot(view, normal));
    const glm::vec3 localHalf = sampleGgxVndfLocal(
        localView, roughness, glm::vec2(randomFloat(rng), randomFloat(rng)));
    return glm::normalize(
        tangent * localHalf.x + bitangent * localHalf.y + normal * localHalf.z);
}

NR_CPU_GPU inline float distributionGgx(
    const glm::vec3 normal, const glm::vec3 halfVector, const float roughness)
{
    const float alpha = roughness * roughness;
    const float alpha2 = alpha * alpha;
    const float ndh = fmaxf(glm::dot(normal, halfVector), 0.0f);
    const float denominator = ndh * ndh * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / fmaxf(BsdfPi * denominator * denominator, BsdfEpsilon);
}

NR_CPU_GPU inline float geometrySchlickGgx(const float nd, const float roughness)
{
    const float k = roughness * roughness * 0.5f;
    return nd / fmaxf(nd * (1.0f - k) + k, BsdfEpsilon);
}

NR_CPU_GPU inline float geometrySmith(
    const glm::vec3 normal, const glm::vec3 view, const glm::vec3 light, const float roughness)
{
    return geometrySchlickGgx(fmaxf(glm::dot(normal, view), 0.0f), roughness)
        * geometrySchlickGgx(fmaxf(glm::dot(normal, light), 0.0f), roughness);
}

NR_CPU_GPU inline float fresnelDielectric(float cosThetaI, float etaI, float etaT)
{
    if (cosThetaI < 0.0f)
    {
        const float swap = etaI;
        etaI = etaT;
        etaT = swap;
        cosThetaI = -cosThetaI;
    }
    const float eta = etaI / etaT;
    const float sin2ThetaT = eta * eta * fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI);
    if (sin2ThetaT >= 1.0f)
        return 1.0f;
    const float cosThetaT = sqrtf(1.0f - sin2ThetaT);
    const float a = etaT * cosThetaI;
    const float b = etaI * cosThetaT;
    const float rs = (a - b) / fmaxf(a + b, BsdfEpsilon);
    const float rpNumerator = a * cosThetaT - b * cosThetaI;
    const float rpDenominator = a * cosThetaT + b * cosThetaI;
    const float rp = rpNumerator / fmaxf(fabsf(rpDenominator), BsdfEpsilon);
    return 0.5f * (rs * rs + rp * rp);
}

NR_CPU_GPU inline glm::vec3 fresnelSchlick(
    const float cosTheta, const glm::vec3 f0)
{
    const float oneMinusCos = 1.0f - fminf(fmaxf(cosTheta, 0.0f), 1.0f);
    const float factor = oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos;
    return f0 + (glm::vec3(1.0f) - f0) * factor;
}

NR_CPU_GPU inline glm::vec3 evaluateGgxReflection(
    const glm::vec3 normal,
    const glm::vec3 view,
    const glm::vec3 light,
    const glm::vec3 fresnel,
    const float roughness,
    const glm::vec3 halfVector)
{
    const float ndv = fmaxf(glm::dot(normal, view), 0.0f);
    const float ndl = fmaxf(glm::dot(normal, light), 0.0f);
    if (ndv <= 0.0f || ndl <= 0.0f)
        return glm::vec3(0.0f);
    const float distribution = distributionGgx(normal, halfVector, roughness);
    const float geometry = geometrySmith(normal, view, light, roughness);
    return fresnel *
        (distribution * geometry / fmaxf(4.0f * ndv * ndl, BsdfEpsilon));
}

NR_CPU_GPU inline glm::vec3 evaluateOpaqueBsdf(
    const glm::vec3 normal,
    const glm::vec3 view,
    const glm::vec3 light,
    const glm::vec3 albedo,
    const float metallic,
    const float specular,
    const float roughness)
{
    const float ndv = fmaxf(glm::dot(normal, view), 0.0f);
    const float ndl = fmaxf(glm::dot(normal, light), 0.0f);
    if (ndv <= 0.0f || ndl <= 0.0f)
        return glm::vec3(0.0f);
    const glm::vec3 halfVector = glm::normalize(view + light);
    const glm::vec3 f0 = glm::mix(
        glm::vec3(fminf(fmaxf(0.08f * specular, 0.0f), 1.0f)), albedo, metallic);
    const glm::vec3 fresnel = fresnelSchlick(
        fmaxf(glm::dot(view, halfVector), 0.0f), f0);
    const glm::vec3 specularBrdf = evaluateGgxReflection(
        normal, view, light, fresnel, roughness, halfVector);
    const glm::vec3 diffuseWeight = (glm::vec3(1.0f) - fresnel) * (1.0f - metallic);
    const glm::vec3 diffuseBrdf = diffuseWeight * albedo / BsdfPi;
    return diffuseBrdf + specularBrdf;
}

NR_CPU_GPU inline glm::vec3 evaluateDirectBsdf(
    const glm::vec3 normal,
    const glm::vec3 view,
    const glm::vec3 light,
    const glm::vec3 albedo,
    const float metallic,
    const float specular,
    const float roughness,
    const float transmission,
    const float ior)
{
    const glm::vec3 opaque = evaluateOpaqueBsdf(
        normal, view, light, albedo, metallic, specular, roughness);
    if (transmission <= 0.0f)
        return opaque;
    if (glm::dot(normal, view) <= 0.0f || glm::dot(normal, light) <= 0.0f)
        return opaque * (1.0f - transmission);
    const glm::vec3 halfVector = glm::normalize(view + light);
    const float dielectricFresnel = fresnelDielectric(
        fmaxf(glm::dot(view, halfVector), 0.0f), 1.0f, fmaxf(ior, 1.0f));
    const glm::vec3 dielectricReflection = evaluateGgxReflection(
        normal, view, light, glm::vec3(dielectricFresnel), roughness, halfVector);
    return opaque * (1.0f - transmission) + dielectricReflection * transmission;
}

NR_CPU_GPU inline float pdfGgxReflection(
    const glm::vec3 view,
    const glm::vec3 normal,
    const glm::vec3 halfVector,
    const float roughness)
{
    const float ndv = fmaxf(glm::dot(normal, view), BsdfEpsilon);
    return geometrySchlickGgx(ndv, roughness)
        * distributionGgx(normal, halfVector, roughness) / (4.0f * ndv);
}

NR_CPU_GPU inline BsdfSample sampleOpaqueBsdf(
    const glm::vec3 view,
    glm::vec3 normal,
    const glm::vec3 albedo,
    const float metallic,
    const float specular,
    const float roughness,
    uint32_t& rng)
{
    if (glm::dot(normal, view) < 0.0f)
        normal = -normal;
    const glm::vec3 dielectricF0(fminf(fmaxf(0.08f * specular, 0.0f), 1.0f));
    const glm::vec3 samplingFresnel = fresnelSchlick(
        fabsf(glm::dot(normal, view)), glm::mix(dielectricF0, albedo, metallic));
    const float specularProbability = fminf(fmaxf(
        fmaxf(samplingFresnel.x, fmaxf(samplingFresnel.y, samplingFresnel.z)),
        0.02f), 0.98f);

    BsdfSample sample{};
    glm::vec3 halfVector{};
    if (randomFloat(rng) < specularProbability)
    {
        sample.event = BsdfEvent::Specular;
        halfVector = sampleGgxHalfVector(view, normal, roughness, rng);
        sample.direction = glm::reflect(-view, halfVector);
    }
    else
    {
        sample.event = BsdfEvent::Diffuse;
        sample.direction = sampleDiffuseDirection(normal, rng);
        halfVector = glm::normalize(view + sample.direction);
    }

    const float ndl = fmaxf(glm::dot(normal, sample.direction), 0.0f);
    if (ndl <= 0.0f)
        return sample;
    const glm::vec3 brdf = evaluateOpaqueBsdf(
        normal, view, sample.direction, albedo, metallic, specular, roughness);
    const float specularPdf = pdfGgxReflection(view, normal, halfVector, roughness);
    const float diffusePdf = ndl / BsdfPi;
    const float pdf = fmaxf(
        specularProbability * specularPdf + (1.0f - specularProbability) * diffusePdf,
        BsdfEpsilon);
    sample.weight = brdf * (ndl / pdf);
    return sample;
}

NR_CPU_GPU inline BsdfSample sampleDielectricBsdf(
    const glm::vec3 view,
    const glm::vec3 geometricNormal,
    const glm::vec3 shadingNormal,
    const float roughness,
    const float ior,
    const glm::vec3 transmissionColor,
    uint32_t& rng)
{
    glm::vec3 orientedShadingNormal = shadingNormal;
    if (glm::dot(orientedShadingNormal, view) < 0.0f)
        orientedShadingNormal = -orientedShadingNormal;
    const glm::vec3 halfVector = sampleGgxHalfVector(
        view, orientedShadingNormal, roughness, rng);
    const float vdh = fmaxf(glm::dot(view, halfVector), 0.0f);

    const glm::vec3 incident = -view;
    const bool exiting = glm::dot(incident, geometricNormal) > 0.0f;
    const float etaI = exiting ? fmaxf(ior, 1.0f) : 1.0f;
    const float etaT = exiting ? 1.0f : fmaxf(ior, 1.0f);
    const float fresnel = fmaxf(fresnelDielectric(vdh, 1.0f, fmaxf(ior, 1.0f)), BsdfEpsilon);
    const glm::vec3 refracted = glm::refract(incident, halfVector, etaI / etaT);
    const bool totalInternalReflection = glm::dot(refracted, refracted) < 1e-10f;

    BsdfSample sample{};
    if (totalInternalReflection || randomFloat(rng) < fresnel)
    {
        sample.event = BsdfEvent::Specular;
        sample.direction = glm::reflect(-view, halfVector);
        const glm::vec3 brdf = evaluateGgxReflection(
            orientedShadingNormal, view, sample.direction,
            glm::vec3(fresnel), roughness, halfVector);
        const float pdf = fmaxf(
            pdfGgxReflection(view, orientedShadingNormal, halfVector, roughness), BsdfEpsilon);
        sample.weight = brdf *
            (fmaxf(glm::dot(orientedShadingNormal, sample.direction), 0.0f) / (pdf * fresnel));
    }
    else
    {
        sample.event = BsdfEvent::Transmission;
        sample.direction = glm::normalize(refracted);
        sample.weight = transmissionColor / fmaxf(1.0f - fresnel, BsdfEpsilon);
    }
    return sample;
}
