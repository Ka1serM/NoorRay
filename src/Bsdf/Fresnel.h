#pragma once

#include <cmath>

#include <glm/vec3.hpp>

#include "Bsdf/Microfacet.h"
#include "CUDA/Annotations.h"
#include "Raytracing/Spectrum.h"

namespace nr::bsdf
{
// Unpolarized Fresnel reflectance for a dielectric interface.
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
    const float rs = (a - b) / fmaxf(a + b, Epsilon);
    const float rpNumerator = a * cosThetaT - b * cosThetaI;
    const float rpDenominator = a * cosThetaT + b * cosThetaI;
    const float rp = rpNumerator / fmaxf(fabsf(rpDenominator), Epsilon);
    return 0.5f * (rs * rs + rp * rp);
}

// Normal-incidence reflectance (F0) expressed as an equivalent dielectric IOR,
// so angular falloff can be evaluated with the real Fresnel equations instead
// of Schlick's power-5 fit.
NR_CPU_GPU inline float iorFromF0(const float f0)
{
    const float clamped = fminf(fmaxf(f0, 0.0f), 0.999f);
    const float sqrtF0 = sqrtf(clamped);
    return (1.0f + sqrtF0) / fmaxf(1.0f - sqrtF0, Epsilon);
}

// Exact dielectric Fresnel reflectance for a surface whose normal-incidence
// reflectance is f0 (a specular-workflow tint, not a physical conductor IOR).
NR_CPU_GPU inline float fresnelFromF0(const float cosTheta, const float f0)
{
    return fresnelDielectric(cosTheta, 1.0f, iorFromF0(f0));
}

NR_CPU_GPU inline glm::vec3 fresnelFromF0(const float cosTheta, const glm::vec3 f0)
{
    return glm::vec3(
        fresnelFromF0(cosTheta, f0.x),
        fresnelFromF0(cosTheta, f0.y),
        fresnelFromF0(cosTheta, f0.z));
}

} // namespace nr::bsdf
