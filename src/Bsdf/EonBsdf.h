#pragma once

#include <cmath>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "Bsdf/Microfacet.h"
#include "CUDA/Annotations.h"
#include "Raytracing/Bsdf.h"
#include "Raytracing/Spectrum.h"
#include "Samplers/HemisphereSampler.h"
#include "Samplers/RandomSampler.h"

namespace nr::bsdf
{

// ---------------------------------------------------------------------------
// Energy-preserving Oren-Nayar (d'Eon 2021) for rough diffuse.
//
// Standard Oren-Nayar single-scatter + analytic multi-scatter compensation
// that restores energy lost at high roughness.  The compensation produces
// an isotropic lobe added to the directional ON component.
//
// Roughness is in [0,1], mapped to sigma = roughness * Pi/2 (radians).
// ---------------------------------------------------------------------------

// Standard Oren-Nayar single-scatter lobe value (no albedo, no cosine).
NR_CPU_GPU inline float orenNayarLobe(
    float ndv,
    float ndl,
    float ndh,
    float vdh,
    float roughness)
{
    const float sigma = roughness * 1.57079632679f; // roughness * Pi/2
    const float sigma2 = sigma * sigma;

    const float A = 1.0f - 0.5f * sigma2 / fmaxf(sigma2 + 0.33f, Epsilon);
    const float B = 0.45f * sigma2 / fmaxf(sigma2 + 0.09f, Epsilon);

    const float cosThetaI = ndv;
    const float cosThetaO = ndl;
    const float thetaI = acosf(fminf(cosThetaI, 1.0f));
    const float thetaO = acosf(fminf(cosThetaO, 1.0f));
    const float alpha = fmaxf(thetaI, thetaO);
    const float beta  = fminf(thetaI, thetaO);

    // cos(Δφ) from the half-vector geometry
    const float cosDeltaPhi = fminf(fmaxf(
        (vdh - ndv * ndl) / fmaxf(
            sqrtf(fmaxf(1.0f - ndv * ndv, 0.0f)) *
            sqrtf(fmaxf(1.0f - ndl * ndl, 0.0f)),
            Epsilon), -1.0f), 1.0f);

    return A + B * fmaxf(cosDeltaPhi, 0.0f) * sinf(alpha) * tanf(beta);
}

// Directional albedo of a Lambertian surface: E(mu) = mu (since we evaluate
// the BRDF without the 1/π factor for energy calculations).
// For Oren-Nayar the directional albedo is roughness-dependent.
NR_CPU_GPU inline float orenNayarDirectionalAlbedo(float ndv, float roughness)
{
    (void)roughness;
    // For the standard Oren-Nayar, the directional-hemispherical reflectance
    // is approximately A + B * (π/2 - θ) * tan(θ) * ... for the directional
    // term.  For the compensator we use a simpler approximation:
    // E(σ, cosθ) ≈ cosθ * (A + B * (1 - cosθ))
    const float sigma = roughness * 1.57079632679f;
    const float sigma2 = sigma * sigma;
    const float A = 1.0f - 0.5f * sigma2 / (sigma2 + 0.33f);
    const float B = 0.45f * sigma2 / (sigma2 + 0.09f);
    const float mu = fminf(fmaxf(ndv, 0.0f), 1.0f);
    return mu * (A + B * (1.0f - mu));
}

// Average (hemispherical-hemispherical) albedo for Oren-Nayar.
NR_CPU_GPU inline float orenNayarAverageAlbedo(float roughness)
{
    const float sigma = roughness * 1.57079632679f;
    const float sigma2 = sigma * sigma;
    const float A = 1.0f - 0.5f * sigma2 / (sigma2 + 0.33f);
    const float B = 0.45f * sigma2 / (sigma2 + 0.09f);
    // Integrated average: 2 * ∫_0^1 mu * (A + B * (1 - mu)) * mu dmu
    // = 2 * (A/3 + B * (1/3 - 1/4))
    // = 2 * (A/3 + B/12)
    return 2.0f * (A / 3.0f + B / 12.0f);
}

// Multi-scatter compensation: isotropic lobe that restores lost energy.
NR_CPU_GPU inline float orenNayarCompensation(
    float ndv,
    float ndl,
    float roughness)
{
    const float eAvg = fminf(fmaxf(orenNayarAverageAlbedo(roughness), 0.0f), 0.999f);
    const float eo = orenNayarDirectionalAlbedo(ndv, roughness);
    const float ei = orenNayarDirectionalAlbedo(ndl, roughness);
    return fmaxf((1.0f - eo) * (1.0f - ei) / fmaxf(Pi * (1.0f - eAvg), Epsilon), 0.0f);
}

// Full EON evaluation: f * cosθ_i.
NR_CPU_GPU inline SampledSpectrum evaluateEonSpectral(
    const glm::vec3 normal,
    const glm::vec3 view,
    const glm::vec3 light,
    const SampledSpectrum& albedo,
    float roughness)
{
    const float ndv = fmaxf(glm::dot(normal, view), 0.0f);
    const float ndl = fmaxf(glm::dot(normal, light), 0.0f);
    if (ndv <= 0.0f || ndl <= 0.0f)
        return SampledSpectrum(0.0f);

    const glm::vec3 halfVector = glm::normalize(view + light);
    const float ndh = fmaxf(glm::dot(normal, halfVector), 0.0f);
    const float vdh = fmaxf(glm::dot(view, halfVector), 0.0f);

    const float onLobe = orenNayarLobe(ndv, ndl, ndh, vdh, roughness);
    const float comp = orenNayarCompensation(ndv, ndl, roughness);

    // EON: (ρ/π) * (lobe + ρ * compensation)
    SampledSpectrum result;
    for (int i = 0; i < NrSpectrumSamples; ++i)
    {
        const float rho = fminf(fmaxf(albedo[i], 0.0f), 1.0f);
        result[i] = (onLobe + rho * comp) * rho / Pi;
    }
    return result;
}

// Sample EON via cosine-weighted hemisphere (the dominant lobe shape).
NR_CPU_GPU inline BsdfSample sampleEonSpectral(
    const glm::vec3 view,
    const glm::vec3 geometricNormal,
    const glm::vec3 shadingNormal,
    const SampledSpectrum& albedo,
    float roughness,
    RandomState& rng)
{
    BsdfSample result{};
    result.albedo = albedo;

    glm::vec3 sampleNormal = shadingNormal;
    if (glm::dot(sampleNormal, view) < 0.0f)
        sampleNormal = -sampleNormal;

    result.event = BsdfEvent::Diffuse;
    result.direction = nr::sampling::cosineHemisphere(
        sampleNormal, glm::vec2(randomFloat(rng), randomFloat(rng)));

    const float ndl = fmaxf(glm::dot(sampleNormal, result.direction), 0.0f);
    if (ndl <= 0.0f)
        return result;

    const float ndv = fmaxf(glm::dot(sampleNormal, view), Epsilon);
    const float pdf = ndl / Pi;
    result.pdf = pdf;

    const glm::vec3 halfVector = glm::normalize(view + result.direction);
    const float ndh = fmaxf(glm::dot(sampleNormal, halfVector), 0.0f);
    const float vdh = fmaxf(glm::dot(view, halfVector), 0.0f);

    const float onLobe = orenNayarLobe(ndv, ndl, ndh, vdh, roughness);
    const float comp = orenNayarCompensation(ndv, ndl, roughness);

    for (int i = 0; i < NrSpectrumSamples; ++i)
    {
        const float rho = fminf(fmaxf(albedo[i], 0.0f), 1.0f);
        const float f = (onLobe + rho * comp) * rho / Pi;
        result.weight[i] = f * (ndl / pdf);
    }

    if (glm::dot(geometricNormal, view) < 0.0f)
    {
        if (glm::dot(geometricNormal, result.direction) <= 0.0f)
            return result;
    }
    else
    {
        if (glm::dot(geometricNormal, result.direction) >= 0.0f)
            return result;
    }
    return result;
}

NR_CPU_GPU inline float pdfEon(
    const glm::vec3 geometricNormal,
    const glm::vec3 normal,
    const glm::vec3 view,
    const glm::vec3 light)
{
    glm::vec3 orientedGeometricNormal = geometricNormal;
    if (glm::dot(orientedGeometricNormal, view) < 0.0f)
        orientedGeometricNormal = -orientedGeometricNormal;
    if (glm::dot(orientedGeometricNormal, light) <= 0.0f)
        return 0.0f;
    const float ndl = fmaxf(glm::dot(normal, light), 0.0f);
    return ndl / Pi;
}

} // namespace nr::bsdf
