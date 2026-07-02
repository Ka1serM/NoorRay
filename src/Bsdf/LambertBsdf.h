#pragma once

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "Raytracing/Bsdf.h"
#include "Samplers/HemisphereSampler.h"
#include "Samplers/RandomSampler.h"

namespace nr::bsdf
{

NR_CPU_GPU inline SampledSpectrum evaluateLambertSpectral(
    const glm::vec3 normal,
    const glm::vec3 view,
    const glm::vec3 light,
    const SampledSpectrum& albedo)
{
    const float ndl = fmaxf(glm::dot(normal, light), 0.0f);
    if (ndl <= 0.0f)
        return SampledSpectrum(0.0f);
    const float ndv = fmaxf(glm::dot(normal, view), 0.0f);
    if (ndv <= 0.0f)
        return SampledSpectrum(0.0f);
    return albedo / Pi;
}

NR_CPU_GPU inline BsdfSample sampleLambertSpectral(
    const glm::vec3 view,
    const glm::vec3 geometricNormal,
    const glm::vec3 shadingNormal,
    const SampledSpectrum& albedo,
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

    const float cosView = fmaxf(glm::dot(sampleNormal, view), 0.0f);
    if (cosView <= 0.0f)
        return result;

    const float pdf = ndl / Pi;
    result.pdf = pdf;

    const SampledSpectrum brdf = albedo / Pi;
    for (int i = 0; i < NrSpectrumSamples; ++i)
        result.weight[i] = brdf[i] * (ndl / pdf);

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

NR_CPU_GPU inline float pdfLambert(
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
