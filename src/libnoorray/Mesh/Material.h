#pragma once

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "CUDA/Annotations.h"
#include "CUDA/Unique/Texture.h"
#include "Mesh/OpenPbrEnergy.h"
#include "Raytracing/Bsdf.h"
#include "Raytracing/RgbToSpectrum.h"
#include "Raytracing/Sellmeier.h"

struct Material
{
public:
    glm::vec3 albedo{1.0f};
    int albedoIndex{-1};
    float specular{1.0f};
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
    // ── Texture resolution (GPU only) ───────────────────────────────────────
    // Resolves this material's textured properties at `uv` and hands the
    // result to a Bsdf, which owns all the actual BSDF math.

    NR_GPU Bsdf makeBsdf(
        const nr::cuda::UniqueTexture* textures,
        const glm::vec2 uv,
        const glm::vec3 view,
        const glm::vec3 geometricNormal,
        const glm::vec3 shadingNormal,
        const SampledWavelengths& wl,
        const float* spectrumScale,
        const float* spectrumCoeffs,
        const nr::openpbr::EnergyLutTextures& openPbrLuts) const
    {
        glm::vec3 rgbAlbedo = albedo;
        if (albedoIndex >= 0)
            rgbAlbedo *= glm::vec3(textures[albedoIndex].sample(uv));
        const SampledSpectrum albedoSpec = rgbAlbedoToSpectrum(
            rgbAlbedo, wl, spectrumScale, spectrumCoeffs);

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

        const SampledSpectrum transmissionColorSpec = rgbAlbedoToSpectrum(
            transmissionColor, wl, spectrumScale, spectrumCoeffs);

        return Bsdf(geometricNormal, shadingNormal, view, albedoSpec, met, spec, rough, trans,
            transmissionColorSpec, sellmeier, wl, openPbrLuts);
    }

    NR_GPU SampledSpectrum emissionSpectral(
        const nr::cuda::UniqueTexture* textures,
        const glm::vec2 uv,
        const SampledWavelengths& wl,
        const float* spectrumScale,
        const float* spectrumCoeffs,
        const float* d65) const
    {
        glm::vec3 rgbEmission = emission;
        if (emissionIndex >= 0)
            rgbEmission *= glm::vec3(textures[emissionIndex].sample(uv));
        rgbEmission *= emissionStrength;
        return rgbIlluminantToSpectrum(rgbEmission, wl, spectrumScale, spectrumCoeffs, d65);
    }
#endif
};
