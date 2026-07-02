#pragma once

#include <cmath>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "Bsdf/ConductorBsdf.h"
#include "Bsdf/DeltaBsdf.h"
#include "Bsdf/DielectricBsdf.h"
#include "Bsdf/EonBsdf.h"
#include "Bsdf/Fresnel.h"
#include "Bsdf/LambertBsdf.h"
#include "Bsdf/Microfacet.h"
#include "CUDA/Annotations.h"
#include "Raytracing/Bsdf.h"
#include "Raytracing/Sellmeier.h"
#include "Raytracing/Spectrum.h"
#include "Samplers/RandomSampler.h"

// Dispatch material model based on parameters:
//
//   transmission > 0:
//     roughness == 0 → delta dielectric (Fresnel reflection + refraction)
//     roughness > 0  → multi-bounce dielectric (coupled reflection/refraction)
//
//   transmission == 0:
//     metallic == 0  → diffuse (Lambert/EON) + specular dielectric reflection
//     metallic == 1  → conductor (delta or multi-bounce)
//     0 < metallic < 1 → blend via stochastic selection

struct BsdfMaterialParameters
{
    SampledSpectrum baseColor{};
    SampledSpectrum transmissionColor{1.0f};
    float metalness{};
    float roughness{};
    float specular{0.5f};
    float transmission{};
    SellmeierCoefficients sellmeier{};
};

namespace nr::bsdf
{

NR_CPU_GPU inline float specularSamplingProbability(
    const glm::vec3 view,
    const glm::vec3 shadingNormal,
    const SampledSpectrum& albedo,
    float metallic,
    float specular)
{
    const float ndv = fabsf(glm::dot(shadingNormal, view));
    const float dielectricF0 = fminf(fmaxf(0.08f * specular, 0.0f), 1.0f);
    float avgAlbedo = 0.0f;
    for (int i = 0; i < NrSpectrumSamples; ++i)
        avgAlbedo += albedo[i];
    avgAlbedo /= static_cast<float>(NrSpectrumSamples);
    const float f0 = dielectricF0 + (avgAlbedo - dielectricF0) * metallic;
    const float samplingFresnel = fresnelDielectric(ndv, 1.0f, iorFromF0(f0));
    return fminf(fmaxf(samplingFresnel, 0.02f), 0.98f);
}

NR_CPU_GPU inline bool hasWavelengthDependentIor(
    const BsdfMaterialParameters& material, const SampledWavelengths& wavelengths)
{
    const float heroIor = sellmeierIor(material.sellmeier, wavelengths[0]);
    for (int i = 1; i < NrSpectrumSamples; ++i)
    {
        if (wavelengths.pdf[i] != 0.0f &&
            fabsf(sellmeierIor(material.sellmeier, wavelengths[i]) - heroIor) > 1e-5f)
            return true;
    }
    return false;
}

// Compute an effective F0 from specular + metallic blend.
NR_CPU_GPU inline float effectiveF0(float specular, float metallic,
    float albedoChannel, float dielectricF0)
{
    return dielectricF0 + (albedoChannel - dielectricF0) * metallic;
}

// ---------------------------------------------------------------------------
// Sample the BSDF.
// ---------------------------------------------------------------------------
NR_CPU_GPU inline BsdfSample sample(
    const BsdfMaterialParameters& material,
    const glm::vec3 geometricNormal,
    const glm::vec3 shadingNormal,
    const glm::vec3 view,
    RandomState& rng,
    const SampledWavelengths& wl)
{
    BsdfSample result{};
    result.albedo = material.baseColor;

    const float roughness = material.roughness;
    const float transmission = material.transmission;
    const float metallic = material.metalness;
    const float specular = material.specular;
    const bool smooth = roughness <= 1e-5f;

    // ---------- Transmissive materials ----------
    if (transmission > 0.0f && randomFloat(rng) < transmission)
    {
        if (smooth)
        {
            // Delta dielectric (smooth Fresnel + refraction)
            const float ior = sellmeierIor(material.sellmeier, wl.lambda[0]);
            result = sampleDeltaDielectric(
                view, geometricNormal, shadingNormal,
                material.baseColor, material.transmissionColor,
                ior, wl, rng);
        }
        else
        {
            // Multi-bounce rough dielectric
            result = sampleDielectricMultiBounce(
                view, geometricNormal, shadingNormal,
                roughness, material.sellmeier, wl,
                material.transmissionColor, rng);
        }

        result.specular = specular;
        result.metallic = metallic;
        result.roughness = roughness;
        result.transmission = transmission;
        return result;
    }

    // ---------- Opaque materials ----------
    // Decide between specular lobe and diffuse lobe for sampling.
    const float specProb = specularSamplingProbability(
        view, shadingNormal, material.baseColor, metallic, specular);

    if (randomFloat(rng) < specProb)
    {
        // Specular sampling branch
        if (metallic > 0.5f)
        {
            // Metal-like: use conductor BSDF
            if (smooth)
            {
                result = sampleDeltaConductor(
                    view, geometricNormal, shadingNormal,
                    material.baseColor, wl, rng);
            }
            else
            {
                result = sampleConductorMultiBounce(
                    view, geometricNormal, shadingNormal,
                    roughness, material.baseColor, wl, rng);
            }
        }
        else
        {
            // Dielectric-like: use artist-friendly F0 for specular
            const float dF0 = fminf(fmaxf(0.08f * specular, 0.0f), 1.0f);
            SampledSpectrum specF0;
            for (int i = 0; i < NrSpectrumSamples; ++i)
                specF0[i] = dF0 + (material.baseColor[i] - dF0) * metallic;

            if (smooth)
            {
                result = sampleDeltaConductor(
                    view, geometricNormal, shadingNormal,
                    specF0, wl, rng);
            }
            else
            {
                // Reflection-only GGX VNDF sampling for opaque dielectrics.
                glm::vec3 orientedNormal = shadingNormal;
                if (glm::dot(orientedNormal, view) < 0.0f)
                    orientedNormal = -orientedNormal;

                const glm::vec3 halfVector = sampleGgxHalfVector(
                    view, orientedNormal, roughness, rng);
                const glm::vec3 reflected = glm::reflect(-view, halfVector);

                result.direction = reflected;
                result.event = BsdfEvent::Specular;
                result.pdf = specProb * pdfGgxReflection(
                    view, orientedNormal, halfVector, roughness);

                const SampledSpectrum fCos = evaluateConductorSingleScatter(
                    orientedNormal, view, reflected, roughness,
                    specF0, wl);
                result.weight = fCos / fmaxf(result.pdf, Epsilon);
            }
        }
    }
    else
    {
        // Diffuse sampling branch
        if (smooth || roughness <= 0.1f)
        {
            result = sampleLambertSpectral(
                view, geometricNormal, shadingNormal,
                material.baseColor, rng);
        }
        else
        {
            result = sampleEonSpectral(
                view, geometricNormal, shadingNormal,
                material.baseColor, roughness, rng);
        }
    }

    result.metallic = metallic;
    result.roughness = roughness;
    result.specular = specular;
    result.transmission = 0.0f;
    return result;
}

// ---------------------------------------------------------------------------
// Evaluate BSDF * cosθ_i for NEE.
// ---------------------------------------------------------------------------
NR_CPU_GPU inline SampledSpectrum evaluate(
    const BsdfMaterialParameters& material,
    const glm::vec3 geometricNormal,
    const glm::vec3 shadingNormal,
    const glm::vec3 view,
    const glm::vec3 light,
    const SampledWavelengths& wl,
    RandomState& rng)
{
    // Orient normals toward the view direction
    glm::vec3 orientedShadingNormal = shadingNormal;
    if (glm::dot(orientedShadingNormal, view) < 0.0f)
        orientedShadingNormal = -orientedShadingNormal;
    glm::vec3 orientedGeometricNormal = geometricNormal;
    if (glm::dot(orientedGeometricNormal, view) < 0.0f)
        orientedGeometricNormal = -orientedGeometricNormal;

    // Reject light from the wrong side of the geometric normal.
    if (glm::dot(orientedGeometricNormal, light) <= 0.0f)
        return SampledSpectrum(0.0f);
    if (glm::dot(orientedShadingNormal, light) <= 0.0f)
        return SampledSpectrum(0.0f);

    const float ndv = fmaxf(glm::dot(orientedShadingNormal, view), Epsilon);
    const float ndl = fmaxf(glm::dot(orientedShadingNormal, light), Epsilon);

    const float roughness = material.roughness;
    const float metallic = material.metalness;
    const float specular = material.specular;
    const float dielectricF0 = fminf(fmaxf(0.08f * specular, 0.0f), 1.0f);

    SampledSpectrum result;

    // Specular contribution (microfacet reflection)
    SampledSpectrum specContrib;
    bool hasSpecular = metallic > 0.0f || specular > 0.0f;

    if (hasSpecular && roughness > 1e-5f)
    {
        if (metallic > 0.5f)
        {
            specContrib = evaluateConductorMultiBounce(
                orientedShadingNormal, view, light, roughness,
                material.baseColor, wl, rng);
        }
        else
        {
            SampledSpectrum specF0;
            for (int i = 0; i < NrSpectrumSamples; ++i)
                specF0[i] = dielectricF0 + (material.baseColor[i] - dielectricF0) * metallic;
            specContrib = evaluateConductorSingleScatter(
                orientedShadingNormal, view, light, roughness,
                specF0, wl);
        }
    }

    SampledSpectrum diffuseContrib;
    const float diffAttenuation = fmaxf(1.0f - metallic, 0.0f);

    if (diffAttenuation > 0.0f)
    {
        if (roughness <= 0.1f)
        {
            diffuseContrib = evaluateLambertSpectral(
                orientedShadingNormal, view, light, material.baseColor);
        }
        else
        {
            diffuseContrib = evaluateEonSpectral(
                orientedShadingNormal, view, light, material.baseColor, roughness);
        }
        diffuseContrib *= diffAttenuation;

        if (roughness > 1e-5f && metallic < 0.5f && specular > 0.0f)
        {
            const float ior = iorFromF0(dielectricF0);
            const float F_in  = fresnelDielectric(ndv, 1.0f, ior);
            const float F_out = fresnelDielectric(ndl, 1.0f, ior);
            diffuseContrib *= (1.0f - F_in) * (1.0f - F_out);
        }
    }

    result = specContrib + diffuseContrib;
    result *= ndl;
    return result;
}

// ---------------------------------------------------------------------------
// PDF for MIS with light sampling.
// ---------------------------------------------------------------------------
NR_CPU_GPU inline float pdf(
    const BsdfMaterialParameters& material,
    const glm::vec3 geometricNormal,
    const glm::vec3 shadingNormal,
    const glm::vec3 view,
    const glm::vec3 light)
{
    const float ndl = fmaxf(glm::dot(shadingNormal, light), 0.0f);
    if (ndl <= 0.0f)
        return 0.0f;

    glm::vec3 orientedGeometricNormal = geometricNormal;
    if (glm::dot(orientedGeometricNormal, view) < 0.0f)
        orientedGeometricNormal = -orientedGeometricNormal;
    if (glm::dot(orientedGeometricNormal, light) <= 0.0f)
        return 0.0f;

    const float roughness = material.roughness;
    const float metallic = material.metalness;
    const float specular = material.specular;
    const float transmission = material.transmission;

    // For transmissive materials, the NEE PDF is zero (can't connect
    // a shadow ray through the surface).
    if (transmission > 0.0f)
        return 0.0f;

    // PDF for specular lobe
    float specPdf = 0.0f;
    const glm::vec3 halfVector = glm::normalize(view + light);
    if (roughness > 1e-5f)
    {
        specPdf = pdfGgxReflection(view, shadingNormal, halfVector, roughness);
    }

    // PDF for diffuse lobe
    const float diffusePdf = ndl / Pi;

    // Mixture probability
    const float specProb = specularSamplingProbability(
        view, shadingNormal, material.baseColor, metallic, specular);

    return specProb * specPdf + (1.0f - specProb) * diffusePdf;
}

} // namespace nr::bsdf
