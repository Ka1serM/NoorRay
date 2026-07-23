#pragma once

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "CUDA/Annotations.h"
#include "CUDA/Unique/Texture.h"
#include "Shading/OpenPbrEnergy.h"
#include "Shading/Bsdf.h"
#include "Shading/RgbToSpectrum.h"
#include "Shading/Sellmeier.h"
#include "Raytracing/Gpu/Types.h"

struct GpuSceneData;
struct Surface;

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

    NR_GPU float opacityAt(
        const nr::cuda::UniqueTexture* textures, const glm::vec2 uv) const
    {
        float value = opacity;
        if (opacityIndex >= 0)
            value *= textures[opacityIndex].sample(uv).w;
        return fminf(fmaxf(value, 0.0f), 1.0f);
    }

    NR_GPU float transmissionAt(
        const nr::cuda::UniqueTexture* textures, const glm::vec2 uv) const
    {
        float value = transmission;
        if (transmissionIndex >= 0)
            value *= textures[transmissionIndex].sample(uv).x;
        return fminf(fmaxf(value, 0.0f), 1.0f);
    }

    NR_GPU glm::vec3 albedoAt(
        const nr::cuda::UniqueTexture* textures, const glm::vec2 uv) const
    {
        glm::vec3 value = albedo;
        if (albedoIndex >= 0)
            value *= glm::vec3(textures[albedoIndex].sample(uv));
        return value;
    }

    NR_GPU bool acceptsRayHit(
        const nr::cuda::UniqueTexture* textures, const glm::vec2 uv,
        RandomState& rng) const
    {
        return randomFloat(rng) <= opacityAt(textures, uv);
    }

    NR_GPU bool blocksShadowRay(
        const nr::cuda::UniqueTexture* textures, const glm::vec2 uv,
        RandomState& rng) const
    {
        const float blockProbability = opacityAt(textures, uv)
            * (1.0f - transmissionAt(textures, uv));
        return blockProbability >= 1.0f || randomFloat(rng) < blockProbability;
    }

    NR_GPU glm::vec3 shadingNormalAt(
        const nr::cuda::UniqueTexture* textures, const glm::vec2 uv,
        const glm::vec3 tangent, const glm::vec3 shadingNormal) const
    {
        if (normalIndex < 0)
            return shadingNormal;
        const glm::vec4 encoded = textures[normalIndex].sample(uv);
        const glm::vec3 tangentNormal = glm::vec3(encoded) * 2.0f - 1.0f;
        const glm::vec3 t = glm::normalize(tangent);
        const glm::vec3 bitangent = glm::normalize(glm::cross(shadingNormal, t));
        return glm::normalize(
            t * tangentNormal.x + bitangent * tangentNormal.y
            + shadingNormal * tangentNormal.z);
    }

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
        const glm::vec3 rgbAlbedo = albedoAt(textures, uv);
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

    // Surface reconstruction belongs to geometry; material owns the conversion
    // of that record into its shading model.
    NR_GPU Bsdf makeBsdf(
        const GpuSceneData& scene,
        const Surface& surface,
        const Ray& incident,
        const glm::vec3& geometricNormal,
        const SampledWavelengths& wavelengths) const;
#endif
};
