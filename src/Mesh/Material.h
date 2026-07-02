#pragma once

#include <cmath>
#include <cstdint>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Bsdf/BsdfMaterial.h"
#include "CUDA/Texture.h"
#include "Raytracing/Bsdf.h"
#include "Raytracing/RgbToSpectrum.h"
#include "Raytracing/Sellmeier.h"
#include "Samplers/RandomSampler.h"

class Material
{
public:
    glm::vec3 albedo{1.0f};
    int albedoIndex{-1};
    float specular{0.5f};
    float metallic{};
    float roughness{};
    SellmeierCoefficients sellmeier{};
    int specularIndex{-1};
    int metallicIndex{-1};
    int roughnessIndex{-1};
    int normalIndex{-1};
    glm::vec3 transmissionColor{1.0f};
    float transmission{};
    glm::vec3 emission{1.0f};
    float emissionStrength{};
    int emissionIndex{-1};
    int transmissionIndex{-1};
    int opacityIndex{-1};
    float opacity{1.0f};

#if defined(NR_GPU_CODE)
    NR_GPU BsdfSample sampleBsdfSpectral(
        const CudaTexture* textures,
        const glm::vec2 uv,
        const glm::vec3 view,
        const glm::vec3 geometricNormal,
        const glm::vec3 shadingNormal,
        RandomState& rng,
        const SampledWavelengths& wl,
        const float* spectrumScale,
        const float* spectrumCoeffs,
        const float* d65) const
    {
        BsdfSample result{};

        // --- Albedo (spectral) ---
        glm::vec3 rgbAlbedo = albedo;
        if (albedoIndex >= 0)
            rgbAlbedo *= glm::vec3(textures[albedoIndex].sample(uv));
        result.albedo = rgbAlbedoToSpectrum(rgbAlbedo, wl, spectrumScale, spectrumCoeffs);

        // --- Emission (spectral) ---
        glm::vec3 rgbEmission = emission;
        if (emissionIndex >= 0)
            rgbEmission *= glm::vec3(textures[emissionIndex].sample(uv));
        rgbEmission *= emissionStrength;
        result.emission = rgbIlluminantToSpectrum(
            rgbEmission, wl, spectrumScale, spectrumCoeffs, d65);

        float met = metallic;
        if (metallicIndex >= 0)
            met *= textures[metallicIndex].sample(uv).z;
        met = fminf(fmaxf(met, 0.0f), 1.0f);

        float rough = roughness;
        if (roughnessIndex >= 0)
            rough *= textures[roughnessIndex].sample(uv).y;
        rough = fminf(fmaxf(rough, 0.0f), 1.0f);

        float trans = transmission;
        if (transmissionIndex >= 0)
            trans *= textures[transmissionIndex].sample(uv).x;
        trans = fminf(fmaxf(trans, 0.0f), 1.0f);

        float spec = specular;
        if (specularIndex >= 0)
            spec *= textures[specularIndex].sample(uv).x;
        spec = fminf(fmaxf(spec, 0.0f), 1.0f);

        const SampledSpectrum transSpec = rgbAlbedoToSpectrum(
            transmissionColor, wl, spectrumScale, spectrumCoeffs);

        const BsdfMaterialParameters parameters{
            result.albedo, transSpec, met, rough, spec, trans, sellmeier};
        const SampledSpectrum sampledEmission = result.emission;
        result = nr::bsdf::sample(parameters, geometricNormal, shadingNormal,
            view, rng, wl);
        result.emission = sampledEmission;
        return result;
    }

    NR_GPU SampledSpectrum evaluateDirectSpectral(
        const BsdfSample& bsdf,
        const glm::vec3 geometricNormal,
        const glm::vec3 normal,
        const glm::vec3 view,
        const glm::vec3 light,
        const SampledWavelengths& wl,
        RandomState& rng) const
    {
        const BsdfMaterialParameters parameters{
            bsdf.albedo, SampledSpectrum(1.0f), bsdf.metallic,
            bsdf.roughness, bsdf.specular, bsdf.transmission, sellmeier};
        return nr::bsdf::evaluate(parameters, geometricNormal, normal, view, light, wl, rng);
    }

    NR_GPU float pdfDirectSpectral(
        const BsdfSample& bsdf,
        const glm::vec3 geometricNormal,
        const glm::vec3 normal,
        const glm::vec3 view,
        const glm::vec3 light) const
    {
        const BsdfMaterialParameters parameters{
            bsdf.albedo, SampledSpectrum(1.0f), bsdf.metallic,
            bsdf.roughness, bsdf.specular, bsdf.transmission, sellmeier};
        return nr::bsdf::pdf(parameters, geometricNormal, normal, view, light);
    }
#endif
};
