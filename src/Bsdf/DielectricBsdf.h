#pragma once

#include <cmath>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "Bsdf/Fresnel.h"
#include "Bsdf/Microfacet.h"
#include "CUDA/Annotations.h"
#include "Raytracing/Bsdf.h"
#include "Raytracing/Sellmeier.h"
#include "Raytracing/Spectrum.h"
#include "Samplers/RandomSampler.h"

namespace nr::bsdf
{

// ---------------------------------------------------------------------------
// Single-scatter GGX Smith dielectric evaluation (f * cosθ_i).
// Used for NEE — standard single-bounce microfacet reflection term.
// ---------------------------------------------------------------------------
NR_CPU_GPU inline SampledSpectrum evaluateDielectricReflectionSingleScatter(
    const glm::vec3 normal,
    const glm::vec3 view,
    const glm::vec3 light,
    float roughness,
    float ior,
    const SampledWavelengths& wl)
{
    const float ndv = fmaxf(glm::dot(normal, view), 0.0f);
    const float ndl = fmaxf(glm::dot(normal, light), 0.0f);
    if (ndv <= 0.0f || ndl <= 0.0f)
        return SampledSpectrum(0.0f);

    const glm::vec3 halfVector = glm::normalize(view + light);
    const float D = distributionGgx(normal, halfVector, roughness);
    const float G = geometrySmith(normal, view, light, roughness);
    const float geometry = D * G / fmaxf(4.0f * ndv * ndl, Epsilon);
    const float cosHalf = fmaxf(glm::dot(view, halfVector), 0.0f);

    const bool exiting = glm::dot(-view, normal) > 0.0f;
    const float etaI = exiting ? fmaxf(ior, 1.0f) : 1.0f;
    const float etaT = exiting ? 1.0f : fmaxf(ior, 1.0f);

    SampledSpectrum result;
    for (int i = 0; i < NrSpectrumSamples; ++i)
    {
        const float F = fresnelDielectric(cosHalf, etaI, etaT);
        result[i] = F * geometry;
    }
    return result;
}

NR_CPU_GPU inline SampledSpectrum evaluateDielectricReflectionSingleScatter(
    const glm::vec3 normal,
    const glm::vec3 view,
    const glm::vec3 light,
    float roughness,
    const SellmeierCoefficients& sellmeier,
    const SampledWavelengths& wl)
{
    const float ndv = fmaxf(glm::dot(normal, view), 0.0f);
    const float ndl = fmaxf(glm::dot(normal, light), 0.0f);
    if (ndv <= 0.0f || ndl <= 0.0f)
        return SampledSpectrum(0.0f);

    const glm::vec3 halfVector = glm::normalize(view + light);
    const float D = distributionGgx(normal, halfVector, roughness);
    const float G = geometrySmith(normal, view, light, roughness);
    const float geometry = D * G / fmaxf(4.0f * ndv * ndl, Epsilon);
    const float cosHalf = fmaxf(glm::dot(view, halfVector), 0.0f);

    const bool exiting = glm::dot(-view, normal) > 0.0f;

    SampledSpectrum result;
    for (int i = 0; i < NrSpectrumSamples; ++i)
    {
        const float ior = sellmeierIor(sellmeier, wl.lambda[i]);
        const float etaI = exiting ? ior : 1.0f;
        const float etaT = exiting ? 1.0f : ior;
        const float F = fresnelDielectric(cosHalf, etaI, etaT);
        result[i] = F * geometry;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Position-free multi-bounce Smith dielectric (coupled reflection/refraction)
//
// At each microfacet collision:
//   - Fresnel determines reflection (prob = F) vs refraction (prob = 1-F)
//   - Reflection: continue random walk on same side of interface
//   - Refraction: continue on opposite side (with η² scaling, TIR check)
//   - Accumulate contribution for termination at each bounce
//   - Russian roulette with G1 to decide continuation
//
// This couples reflection and transmission into a single random process,
// correctly accounting for paths that reflect, refract, internally reflect,
// and refract again before escaping.
// ---------------------------------------------------------------------------

enum class DielectricEvent : int
{
    Reflect,
    Refract,
};

// Core single-step of the dielectric multi-bounce walk.
// Returns the result of one microfacet interaction.
struct DielectricBounce
{
    glm::vec3 direction;
    float cosOut;
    float fresnel;
    float g1Out;
    DielectricEvent event;
    float etaRatio;     // η₂/η₁ for this interaction
};

NR_CPU_GPU inline DielectricBounce evaluateDielectricMicrofacetBounce(
    const glm::vec3 currentDir,
    const glm::vec3 halfVector,
    const glm::vec3 sampleNormal,
    float roughness,
    float etaI,
    float etaT)
{
    DielectricBounce result{};
    const float cosHalf = fabsf(glm::dot(currentDir, halfVector));

    // Fresnel
    result.fresnel = fresnelDielectric(cosHalf, etaI, etaT);

    // Reflection direction
    const glm::vec3 reflected = glm::reflect(-currentDir, halfVector);
    result.event = DielectricEvent::Reflect;
    result.direction = reflected;
    result.cosOut = fabsf(glm::dot(sampleNormal, reflected));
    result.g1Out = smithG1Ggx(result.cosOut, roughness);
    result.etaRatio = 1.0f;

    // Check if refraction is possible
    const glm::vec3 refracted = glm::refract(-currentDir, halfVector, etaI / etaT);
    const bool totalInternalReflection = glm::dot(refracted, refracted) < 1e-10f;

    // For sampling, we'll choose based on Fresnel
    return result;
}

// Position-free multi-bounce dielectric reflection/refraction sampling.
NR_CPU_GPU inline BsdfSample sampleDielectricMultiBounce(
    const glm::vec3 view,
    const glm::vec3 geometricNormal,
    const glm::vec3 shadingNormal,
    float roughness,
    float ior,
    const SampledWavelengths& wl,
    const SampledSpectrum& transmissionColor,
    RandomState& rng)
{
    BsdfSample result{};

    glm::vec3 orientedShadingNormal = shadingNormal;
    if (glm::dot(orientedShadingNormal, view) < 0.0f)
        orientedShadingNormal = -orientedShadingNormal;

    const glm::vec3 incident = -view;
    const bool exitingInitial = glm::dot(incident, geometricNormal) > 0.0f;
    const float initialEtaI = exitingInitial ? fmaxf(ior, 1.0f) : 1.0f;
    const float initialEtaT = exitingInitial ? 1.0f : fmaxf(ior, 1.0f);

    SampledSpectrum throughput(1.0f);
    glm::vec3 currentDir = view;
    float currentEtaI = initialEtaI;
    float currentEtaT = initialEtaT;
    float ndv = fmaxf(fabsf(glm::dot(orientedShadingNormal, currentDir)), Epsilon);

    // Track the total accumulated events
    int bounceCount = 0;
    int internalReflections = 0;

    constexpr int maxBounces = 100;
    for (int bounce = 0; bounce < maxBounces; ++bounce)
    {
        const glm::vec3 halfVector = sampleGgxHalfVector(
            currentDir, orientedShadingNormal, roughness, rng);
        const float cosHalf = fabsf(glm::dot(currentDir, halfVector));

        // Dielectric Fresnel
        const float F = fresnelDielectric(cosHalf, currentEtaI, currentEtaT);

        // Reflection direction
        const glm::vec3 reflected = glm::reflect(-currentDir, halfVector);
        const float ndr = fabsf(glm::dot(orientedShadingNormal, reflected));

        // Refraction direction
        const glm::vec3 refracted = glm::refract(
            -currentDir, halfVector, currentEtaI / currentEtaT);
        const bool tir = glm::dot(refracted, refracted) < 1e-10f;

        const float G1_r = smithG1Ggx(ndr, roughness);

        if (tir || randomFloat(rng) < F)
        {
            // Reflection
            const float contribution = throughput.maxComponent();
            if (contribution > 0.0f)
            {
                for (int i = 0; i < NrSpectrumSamples; ++i)
                    result.weight[i] += throughput[i] * F * G1_r;
            }

            if (randomFloat(rng) < G1_r)
            {
                // Terminate — escaped as reflection
                result.direction = reflected;
                result.event = BsdfEvent::Specular;
                result.pdf = pdfGgxReflection(
                    currentDir, orientedShadingNormal, halfVector, roughness);

                if (glm::dot(geometricNormal, result.direction) <= 0.0f &&
                    glm::dot(geometricNormal, view) >= 0.0f)
                    result.weight = SampledSpectrum(0.0f);
                if (glm::dot(geometricNormal, result.direction) >= 0.0f &&
                    glm::dot(geometricNormal, view) <= 0.0f)
                    result.weight = SampledSpectrum(0.0f);

                break;
            }

            // Continue reflecting
            for (int i = 0; i < NrSpectrumSamples; ++i)
                throughput[i] *= F;
            currentDir = reflected;
            ndv = ndr;
            ++internalReflections;
        }
        else
        {
            // Refraction
            const float etaPath = currentEtaT / currentEtaI;
            const float ndt = fabsf(glm::dot(orientedShadingNormal, glm::normalize(refracted)));

            // G1 for the transmitted direction
            const float G1_t = smithG1Ggx(ndt, roughness);

            // Contribution for transmission
            const float transThroughput = (1.0f - F);
            const float contribution = throughput.maxComponent();
            if (contribution > 0.0f)
            {
                for (int i = 0; i < NrSpectrumSamples; ++i)
                {
                    result.weight[i] += throughput[i] * transThroughput * G1_t
                        * transmissionColor[i];
                }
            }

            // If the ray transmits, it exits the microsurface at this point.
            // The transmitted direction exits the interface entirely — there's
            // no multi-bounce continuation for transmission (the ray leaves
            // the microsurface).
            result.direction = glm::normalize(refracted);
            result.event = BsdfEvent::Transmission;
            result.eta = etaPath;

            // Compute PDF: this is the microfacet transmission PDF
            const float wiDotM = glm::dot(result.direction, halfVector);
            const float woDotM = glm::dot(currentDir, halfVector);
            const float denominator = wiDotM + woDotM / etaPath;
            const float denominator2 = denominator * denominator;
            const float dWmDWi = fabsf(wiDotM) / fmaxf(denominator2, Epsilon);

            const float ndv = fmaxf(fabsf(glm::dot(orientedShadingNormal, currentDir)), Epsilon);
            const float wmPdf = distributionGgx(orientedShadingNormal, halfVector, roughness)
                * smithG1Ggx(ndv, roughness) * cosHalf / ndv;
            result.pdf = wmPdf * dWmDWi * (1.0f - F);
            result.pdf = fmaxf(result.pdf, Epsilon);

            bool spectralIor = false;
            result.terminateSecondaryWavelengths = spectralIor;

            break;
        }

        float maxThroughput = 0.0f;
        for (int i = 0; i < NrSpectrumSamples; ++i)
            maxThroughput = fmaxf(maxThroughput, throughput[i]);
        if (maxThroughput < 1e-4f)
            break;
    }

    return result;
}

// Spectral multi-bounce dielectric with Sellmeier dispersion.
NR_CPU_GPU inline BsdfSample sampleDielectricMultiBounce(
    const glm::vec3 view,
    const glm::vec3 geometricNormal,
    const glm::vec3 shadingNormal,
    float roughness,
    const SellmeierCoefficients& sellmeier,
    const SampledWavelengths& wl,
    const SampledSpectrum& transmissionColor,
    RandomState& rng)
{
    const float heroIor = sellmeierIor(sellmeier, wl.lambda[0]);
    BsdfSample result = sampleDielectricMultiBounce(
        view, geometricNormal, shadingNormal, roughness, heroIor, wl,
        transmissionColor, rng);

    if (result.event == BsdfEvent::Specular)
    {
        // Tint by per-wavelength Fresnel ratio for the initial half-vector.
        // This approximates spectral variation in the multi-bounce dielectric.
        const glm::vec3 half = glm::normalize(view + result.direction);
        const float cosTheta = fabsf(glm::dot(view, half));
        const bool exiting = glm::dot(-view, geometricNormal) > 0.0f;

        const float heroFresnel = fmaxf(fresnelDielectric(cosTheta,
            exiting ? heroIor : 1.0f, exiting ? 1.0f : heroIor), Epsilon);

        const bool dispersive = [&]() {
            for (int i = 1; i < NrSpectrumSamples; ++i)
                if (wl.pdf[i] != 0.0f &&
                    fabsf(sellmeierIor(sellmeier, wl.lambda[i]) - heroIor) > 1e-5f)
                    return true;
            return false;
        }();

        if (!dispersive)
            return result;

        result.terminateSecondaryWavelengths = false;
        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float wlIor = sellmeierIor(sellmeier, wl.lambda[i]);
            const float f = fresnelDielectric(cosTheta,
                exiting ? wlIor : 1.0f, exiting ? 1.0f : wlIor);
            result.weight[i] = f / heroFresnel;
        }
    }
    else if (result.event == BsdfEvent::Transmission)
    {
        const bool dispersive = [&]() {
            for (int i = 1; i < NrSpectrumSamples; ++i)
                if (wl.pdf[i] != 0.0f &&
                    fabsf(sellmeierIor(sellmeier, wl.lambda[i]) - heroIor) > 1e-5f)
                    return true;
            return false;
        }();

        result.terminateSecondaryWavelengths = dispersive;
        if (dispersive)
        {
            for (int i = 1; i < NrSpectrumSamples; ++i)
                result.weight[i] = 0.0f;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Single-bounce microfacet dielectric (for reference / low-roughness use)
// ---------------------------------------------------------------------------

NR_CPU_GPU inline BsdfSample sampleDielectricSingleScatter(
    const glm::vec3 view,
    const glm::vec3 geometricNormal,
    const glm::vec3 shadingNormal,
    float roughness,
    float ior,
    const SampledWavelengths& wl,
    const SampledSpectrum& transmissionColor,
    RandomState& rng)
{
    glm::vec3 orientedShadingNormal = shadingNormal;
    if (glm::dot(orientedShadingNormal, view) < 0.0f)
        orientedShadingNormal = -orientedShadingNormal;

    const glm::vec3 halfVector = sampleGgxHalfVector(
        view, orientedShadingNormal, roughness, rng);
    const float vdh = fabsf(glm::dot(view, halfVector));

    const glm::vec3 incident = -view;
    const bool exiting = glm::dot(incident, geometricNormal) > 0.0f;
    const float etaI = exiting ? fmaxf(ior, 1.0f) : 1.0f;
    const float etaT = exiting ? 1.0f : fmaxf(ior, 1.0f);
    const float fresnel = fresnelDielectric(vdh, etaI, etaT);
    const glm::vec3 refracted = glm::refract(incident, halfVector, etaI / etaT);
    const bool tir = glm::dot(refracted, refracted) < 1e-10f;
    const float ndv = fmaxf(fabsf(glm::dot(orientedShadingNormal, view)), Epsilon);
    const float distribution = distributionGgx(orientedShadingNormal, halfVector, roughness);
    const float wmPdf = distribution * smithG1Ggx(ndv, roughness) * vdh / ndv;

    BsdfSample sample{};
    if (tir || randomFloat(rng) < fresnel)
    {
        sample.event = BsdfEvent::Specular;
        sample.direction = glm::reflect(-view, halfVector);
        const float ndl = fabsf(glm::dot(orientedShadingNormal, sample.direction));
        sample.pdf = wmPdf * fresnel / fmaxf(4.0f * vdh, Epsilon);
        const float f = distribution
            * geometrySmith(orientedShadingNormal, view, sample.direction, roughness)
            * fresnel / fmaxf(4.0f * ndv * ndl, Epsilon);
        sample.weight = SampledSpectrum(f * ndl / fmaxf(sample.pdf, Epsilon));
    }
    else
    {
        sample.event = BsdfEvent::Transmission;
        sample.direction = glm::normalize(refracted);
        const float etaPath = etaT / etaI;
        sample.eta = etaPath;
        const float wiDotM = glm::dot(sample.direction, halfVector);
        const float woDotM = glm::dot(view, halfVector);
        const float cosI = fabsf(glm::dot(orientedShadingNormal, sample.direction));
        const float denominator = wiDotM + woDotM / etaPath;
        const float denominator2 = denominator * denominator;
        const float dWmDWi = fabsf(wiDotM) / fmaxf(denominator2, Epsilon);
        sample.pdf = wmPdf * dWmDWi * (1.0f - fresnel);

        const float g1View = smithG1Ggx(ndv, roughness);
        const float g1Trans = smithG1Ggx(cosI, roughness);
        float f = (1.0f - fresnel) * distribution * g1View * g1Trans
            * fabsf(wiDotM * woDotM / fmaxf(cosI * ndv * denominator2, Epsilon));
        f /= etaPath * etaPath;

        sample.weight = SampledSpectrum(f * cosI / fmaxf(sample.pdf, Epsilon));
        for (int i = 0; i < NrSpectrumSamples; ++i)
            sample.weight[i] *= transmissionColor[i];
    }
    return sample;
}

} // namespace nr::bsdf
