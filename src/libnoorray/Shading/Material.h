#pragma once

#include <cstdint>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "CUDA/Annotations.h"
#include "CUDA/Unique/Texture.h"
#include "Shading/RgbToSpectrum.h"
#include "Shading/Sellmeier.h"
#include "Raytracing/Gpu/Types.h"

struct GpuSceneData;
struct Surface;

// Host-side authoring state. This is never placed in GpuSceneData or copied
// to the device; it is consumed only while constructing a MaterialX document.
struct MaterialAuthoring
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

    bool hasDispersiveIor(
        const SampledWavelengths& wavelengths) const
    {
        if (wavelengths.secondaryTerminated())
            return false;

        const float heroIor = sellmeierIor(sellmeier, wavelengths[0]);
        for (int i = 1; i < NrSpectrumSamples; ++i)
            if (sellmeierIor(sellmeier, wavelengths[i]) != heroIor)
                return true;
        return false;
    }
};

// Compact GPU-visible material. Every material is an SVM program; the four
// spans address the scene's contiguous instruction and texture-index buffers.
struct Material
{
    std::uint32_t svmBytecodeOffset{};
    std::uint32_t svmBytecodeLength{};
    std::uint32_t svmTextureOffset{};
    std::uint32_t svmTextureCount{};
};
