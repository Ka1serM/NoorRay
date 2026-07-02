#pragma once

#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "CUDA/Texture.h"
#include "Mesh/Material.h"
#include "Raytracing/Bsdf.h"
#include "Raytracing/Sellmeier.h"
#include "Raytracing/Spectrum.h"

#include "Samplers/RandomSampler.h"

#include "OpenPbr/OpenPbrInputs.h"

namespace nr::openpbr {

// ── Convert OpenPBR's 3-channel RGB output to a SampledSpectrum ──────────
// Since all 3 OpenPBR channels use the same hero wavelength, the RGB triple
// IS the BSDF's spectral response at that wavelength, expressed in sRGB
// primaries.  We map R→λ[0], G→λ[1], B→λ[2] and average→λ[3] so each
// wavelength sample gets a plausible scalar weight.
NR_GPU inline SampledSpectrum rgbToSampledSpectrum(glm::vec3 rgb)
{
    const float avg = (rgb.x + rgb.y + rgb.z) * (1.0f / 3.0f);
    SampledSpectrum s;
    s.values[0] = rgb.x;
    s.values[1] = rgb.y;
    s.values[2] = rgb.z;
    s.values[3] = avg;
    return s;
}

// ── SampledSpectrum → vec3 (first 3 channels for path_throughput) ─────────
NR_GPU inline vec3 throughputToVec3(const SampledSpectrum& s)
{
    return vec3(s.values[0], s.values[1], s.values[2]);
}

// ── IOR at a given wavelength via sellmeier ───────────────────────────────
NR_GPU inline float computeIor(const Material& material, float lambdaNm)
{
    return sellmeierIor(material.sellmeier, lambdaNm);
}

// Merge diffuse+specular into a single vec3.
NR_GPU inline glm::vec3 totalWeight(const OpenPBR_DiffuseSpecular& w)
{
    return glm::vec3(
        w.diffuse.x + w.specular.x,
        w.diffuse.y + w.specular.y,
        w.diffuse.z + w.specular.z);
}

// ── Main sample function ─────────────────────────────────────────────────
NR_GPU inline BsdfSample sample(
    const Material& material,
    const CudaTexture* textures,
    glm::vec2 uv,
    glm::vec3 viewDirection,
    glm::vec3 shadingNormal,
    glm::vec3 tangent,
    RandomState& rng,
    const SampledWavelengths& wl,
    float exteriorIor,
    const SampledSpectrum& throughput)
{
    BsdfSample result{};

    const float ior = computeIor(material, wl.lambda[0]);

    const OpenPBR_ResolvedInputs inputs = buildInputs(
        material, textures, uv, shadingNormal, tangent, 1.0f, ior);

    const vec3 throughput_rgb = throughputToVec3(throughput);
    const vec3 rgb_wavelengths_nm(wl.lambda[0], wl.lambda[0], wl.lambda[0]);

    OpenPBR_PreparedBsdf prepared = openpbr_prepare(
        inputs, throughput_rgb, rgb_wavelengths_nm, exteriorIor,
        vec3(viewDirection.x, viewDirection.y, viewDirection.z));

    vec3 rand3 = vec3(randomFloat(rng), randomFloat(rng), randomFloat(rng));
    vec3 lightDirection;
    OpenPBR_DiffuseSpecular weight;
    float pdf;
    OpenPBR_BsdfLobeType sampledType;
    openpbr_sample(prepared, rand3, lightDirection, weight, pdf, sampledType);

    result.weight = rgbToSampledSpectrum(totalWeight(weight));
    result.direction = glm::vec3(lightDirection.x, lightDirection.y, lightDirection.z);
    result.pdf = pdf;
    result.eta = 1.0f;

    const glm::vec3 emissionRgb(
        prepared.emission.x, prepared.emission.y, prepared.emission.z);
    if (emissionRgb.x > 0.0f || emissionRgb.y > 0.0f || emissionRgb.z > 0.0f)
        result.emission = rgbToSampledSpectrum(emissionRgb);

    if (sampledType & OpenPBR_BsdfLobeTypeTransmission)
        result.event = BsdfEvent::Transmission;
    else if (sampledType & OpenPBR_BsdfLobeTypeSpecular)
        result.event = BsdfEvent::Specular;
    else
        result.event = BsdfEvent::Diffuse;

    return result;
}

// ── Evaluate BSDF at given light direction (for NEE) ─────────────────────
NR_GPU inline SampledSpectrum evaluate(
    const Material& material,
    const CudaTexture* textures,
    glm::vec2 uv,
    glm::vec3 viewDirection,
    glm::vec3 lightDirection,
    glm::vec3 shadingNormal,
    glm::vec3 tangent,
    const SampledWavelengths& wl,
    float exteriorIor,
    const SampledSpectrum& throughput)
{
    const float ior = computeIor(material, wl.lambda[0]);

    const OpenPBR_ResolvedInputs inputs = buildInputs(
        material, textures, uv, shadingNormal, tangent, 1.0f, ior);

    const vec3 throughput_rgb = throughputToVec3(throughput);
    const vec3 rgb_wavelengths_nm(wl.lambda[0], wl.lambda[0], wl.lambda[0]);

    OpenPBR_PreparedBsdf prepared = openpbr_prepare(
        inputs, throughput_rgb, rgb_wavelengths_nm, exteriorIor,
        vec3(viewDirection.x, viewDirection.y, viewDirection.z));

    OpenPBR_DiffuseSpecular w = openpbr_eval(
        prepared, vec3(lightDirection.x, lightDirection.y, lightDirection.z));

    return rgbToSampledSpectrum(totalWeight(w));
}

// ── BSDF PDF at given light direction ────────────────────────────────────
NR_GPU inline float pdf(
    const Material& material,
    const CudaTexture* textures,
    glm::vec2 uv,
    glm::vec3 viewDirection,
    glm::vec3 lightDirection,
    glm::vec3 shadingNormal,
    glm::vec3 tangent,
    const SampledSpectrum& throughput)
{
    const float ior = computeIor(material, 580.0f);

    const OpenPBR_ResolvedInputs inputs = buildInputs(
        material, textures, uv, shadingNormal, tangent, 1.0f, ior);

    const vec3 throughput_rgb = throughputToVec3(throughput);
    const vec3 rgb_wavelengths_nm(580.0f, 580.0f, 580.0f);

    OpenPBR_PreparedBsdf prepared = openpbr_prepare(
        inputs, throughput_rgb, rgb_wavelengths_nm, 1.0f,
        vec3(viewDirection.x, viewDirection.y, viewDirection.z));

    return openpbr_pdf(prepared, vec3(lightDirection.x, lightDirection.y, lightDirection.z));
}

} // namespace nr::openpbr
