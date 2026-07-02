#pragma once

#include <cmath>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "Bsdf/Fresnel.h"
#include "CUDA/Annotations.h"
#include "Raytracing/Bsdf.h"
#include "Raytracing/Spectrum.h"
#include "Samplers/RandomSampler.h"

namespace nr::bsdf
{

// Smooth (delta) conductor: perfect Fresnel-weighted reflection.
// Uses fresnelFromF0 with the base color as the normal-incidence reflectance.
NR_CPU_GPU inline BsdfSample sampleDeltaConductor(
    const glm::vec3 view,
    const glm::vec3 geometricNormal,
    const glm::vec3 shadingNormal,
    const SampledSpectrum& albedo,
    const SampledWavelengths& wl,
    RandomState& rng)
{
    (void)rng;
    BsdfSample result{};
    result.albedo = albedo;

    glm::vec3 sampleNormal = shadingNormal;
    if (glm::dot(sampleNormal, view) < 0.0f)
        sampleNormal = -sampleNormal;

    const float ndv = fmaxf(glm::dot(sampleNormal, view), Epsilon);
    result.direction = glm::reflect(-view, sampleNormal);
    result.event = BsdfEvent::Specular;
    result.pdf = 1.0f;

    for (int i = 0; i < NrSpectrumSamples; ++i)
    {
        const float F = fresnelFromF0(ndv, albedo[i]);
        result.weight[i] = albedo[i] * F;
    }

    return result;
}

// Smooth (delta) dielectric: Fresnel-weighted reflection or refraction.
NR_CPU_GPU inline BsdfSample sampleDeltaDielectric(
    const glm::vec3 view,
    const glm::vec3 geometricNormal,
    const glm::vec3 shadingNormal,
    const SampledSpectrum& albedo,
    const SampledSpectrum& transmissionColor,
    float ior,
    const SampledWavelengths& wl,
    RandomState& rng)
{
    BsdfSample result{};
    result.albedo = albedo;

    glm::vec3 orientedShadingNormal = shadingNormal;
    if (glm::dot(orientedShadingNormal, view) < 0.0f)
        orientedShadingNormal = -orientedShadingNormal;

    const glm::vec3 incident = -view;
    const bool exiting = glm::dot(incident, geometricNormal) > 0.0f;
    const float etaI = exiting ? fmaxf(ior, 1.0f) : 1.0f;
    const float etaT = exiting ? 1.0f : fmaxf(ior, 1.0f);
    const float ndv = fmaxf(glm::dot(orientedShadingNormal, view), Epsilon);
    const float fresnel = fresnelDielectric(ndv, etaI, etaT);

    const glm::vec3 refracted = glm::refract(incident, orientedShadingNormal, etaI / etaT);
    const bool totalInternalReflection = glm::dot(refracted, refracted) < 1e-10f;

    result.pdf = 1.0f;

    if (totalInternalReflection || randomFloat(rng) < fresnel)
    {
        result.event = BsdfEvent::Specular;
        result.direction = glm::reflect(-view, orientedShadingNormal);
        const SampledSpectrum F = SampledSpectrum(fresnel);
        for (int i = 0; i < NrSpectrumSamples; ++i)
            result.weight[i] = albedo[i] * F[i];
    }
    else
    {
        result.event = BsdfEvent::Transmission;
        result.direction = glm::normalize(refracted);
        const float etaPath = etaT / etaI;
        result.eta = etaPath;

        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float trans = (1.0f - fresnel);
            result.weight[i] = transmissionColor[i] * trans;
        }

        const bool spectralIor = false;
        result.terminateSecondaryWavelengths = spectralIor;
    }

    return result;
}

NR_CPU_GPU inline float pdfDelta()
{
    return 1.0f;
}

} // namespace nr::bsdf
