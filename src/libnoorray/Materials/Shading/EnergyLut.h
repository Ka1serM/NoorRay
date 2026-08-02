#pragma once

#include <cstddef>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "Backend/CUDA/Annotations.h"
#include "Backend/CUDA/Unique/Texture.h"
#include "Materials/Shading/EnergyLut/EnergyLutConfig.h"

namespace nr::shading::energy_lut
{

namespace data
{

inline constexpr uint16_t GgxDirectionalAlbedo[
    GgxCosineSize * GgxRoughnessSize] = {
#include "Materials/Shading/EnergyLut/Generated/ggx_directional_albedo.h"
};

inline constexpr uint16_t GgxAverageAlbedo[GgxRoughnessSize] = {
#include "Materials/Shading/EnergyLut/Generated/ggx_average_albedo.h"
};

inline constexpr uint16_t DielectricDirectionalAlbedo[
    DielectricCosineSize * DielectricRoughnessSize * DielectricF0Size] = {
#include "Materials/Shading/EnergyLut/Generated/dielectric_directional_albedo.h"
};

inline constexpr uint16_t DielectricAverageAlbedo[
    DielectricRoughnessSize * DielectricF0Size] = {
#include "Materials/Shading/EnergyLut/Generated/dielectric_average_albedo.h"
};

inline constexpr uint16_t GlassDirectionalAlbedo[
    GlassCosineSize * GlassRoughnessSize * GlassIorSize] = {
#include "Materials/Shading/EnergyLut/Generated/glass_directional_albedo.h"
};

inline constexpr uint16_t GlassAverageAlbedo[
    GlassRoughnessSize * GlassIorSize] = {
#include "Materials/Shading/EnergyLut/Generated/glass_average_albedo.h"
};

}

struct Textures
{
    cudaTextureObject_t ggxDirectional{};
    cudaTextureObject_t ggxAverage{};
    cudaTextureObject_t dielectricDirectional{};
    cudaTextureObject_t dielectricAverage{};
    cudaTextureObject_t glassDirectional{};
    cudaTextureObject_t glassAverage{};
};

class Storage
{
public:
    Textures upload(cudaStream_t stream);
    Textures textures() const noexcept;

private:
    nr::cuda::UniqueTexture ggxDirectional_;
    nr::cuda::UniqueTexture ggxAverage_;
    nr::cuda::UniqueTexture dielectricDirectional_;
    nr::cuda::UniqueTexture dielectricAverage_;
    nr::cuda::UniqueTexture glassDirectional_;
    nr::cuda::UniqueTexture glassAverage_;
};

NR_CPU_GPU inline float clampUnit(const float value)
{
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

NR_CPU_GPU inline float decode(const uint16_t value)
{
    return static_cast<float>(value) * (1.0f / 65535.0f);
}

NR_CPU_GPU inline float lerp(
    const float a, const float b, const float amount)
{
    return a + (b - a) * amount;
}

NR_CPU_GPU inline float sample2dHost(
    const uint16_t* values,
    const int width,
    const int height,
    const float x,
    const float y)
{
    const float px = clampUnit(x) * static_cast<float>(width - 1);
    const float py = clampUnit(y) * static_cast<float>(height - 1);
    const int x0 = static_cast<int>(floorf(px));
    const int y0 = static_cast<int>(floorf(py));
    const int x1 = x0 < width - 1 ? x0 + 1 : x0;
    const int y1 = y0 < height - 1 ? y0 + 1 : y0;
    const float tx = px - static_cast<float>(x0);
    const float ty = py - static_cast<float>(y0);
    return lerp(
        lerp(decode(values[y0 * width + x0]),
            decode(values[y0 * width + x1]), tx),
        lerp(decode(values[y1 * width + x0]),
            decode(values[y1 * width + x1]), tx), ty);
}

NR_CPU_GPU inline float sample3dHost(
    const uint16_t* values,
    const int width,
    const int height,
    const int depth,
    const float x,
    const float y,
    const float z)
{
    const float pz = clampUnit(z) * static_cast<float>(depth - 1);
    const int z0 = static_cast<int>(floorf(pz));
    const int z1 = z0 < depth - 1 ? z0 + 1 : z0;
    const float tz = pz - static_cast<float>(z0);
    const size_t plane = static_cast<size_t>(width) * height;
    return lerp(
        sample2dHost(values + static_cast<size_t>(z0) * plane,
            width, height, x, y),
        sample2dHost(values + static_cast<size_t>(z1) * plane,
            width, height, x, y), tz);
}

NR_CPU_GPU inline float sample2d(
    const cudaTextureObject_t texture,
    const uint16_t* hostValues,
    const int width,
    const int height,
    const float x,
    const float y)
{
#if defined(__CUDA_ARCH__)
    return tex2D<float>(texture,
        clampUnit(x) * static_cast<float>(width - 1) + 0.5f,
        clampUnit(y) * static_cast<float>(height - 1) + 0.5f);
#else
    (void)texture;
    return sample2dHost(hostValues, width, height, x, y);
#endif
}

NR_CPU_GPU inline float sample3d(
    const cudaTextureObject_t texture,
    const uint16_t* hostValues,
    const int width,
    const int height,
    const int depth,
    const float x,
    const float y,
    const float z)
{
#if defined(__CUDA_ARCH__)
    return tex3D<float>(texture,
        clampUnit(x) * static_cast<float>(width - 1) + 0.5f,
        clampUnit(y) * static_cast<float>(height - 1) + 0.5f,
        clampUnit(z) * static_cast<float>(depth - 1) + 0.5f);
#else
    (void)texture;
    return sample3dHost(hostValues, width, height, depth, x, y, z);
#endif
}

NR_CPU_GPU inline float cosineCoordinate(const float cosine)
{
    return sqrtf(clampUnit(cosine));
}

NR_CPU_GPU inline float dielectricF0Coordinate(const float f0)
{
    return sqrtf(clampUnit(f0 / MaximumDielectricF0));
}

NR_CPU_GPU inline float glassIorCoordinate(const float relativeIor)
{
    const float eta = fmaxf(relativeIor, 1.0e-5f);
    const bool exiting = eta < 1.0f;
    const float ior = exiting ? 1.0f / eta : eta;
    const float z = sqrtf(fmaxf((ior - 1.0f) / (ior + 1.0f), 0.0f));
    const float side = clampUnit(z / MaximumGlassZ);
    const float index = exiting
        ? static_cast<float>(GlassIorHalfSize - 1) * (1.0f - side)
        : static_cast<float>(GlassIorHalfSize)
            + static_cast<float>(GlassIorHalfSize - 1) * side;
    return index / static_cast<float>(GlassIorSize - 1);
}

NR_CPU_GPU inline float ggxDirectionalAlbedo(
    const Textures* textures, const float roughness, const float cosine)
{
#if defined(__CUDA_ARCH__)
    const uint16_t* hostValues = nullptr;
#else
    const uint16_t* hostValues = data::GgxDirectionalAlbedo;
#endif
    return sample2d(textures ? textures->ggxDirectional : 0, hostValues,
        GgxCosineSize, GgxRoughnessSize,
        cosineCoordinate(cosine), clampUnit(roughness));
}

NR_CPU_GPU inline float ggxAverageAlbedo(
    const Textures* textures, const float roughness)
{
#if defined(__CUDA_ARCH__)
    const uint16_t* hostValues = nullptr;
#else
    const uint16_t* hostValues = data::GgxAverageAlbedo;
#endif
    return sample2d(textures ? textures->ggxAverage : 0, hostValues,
        GgxRoughnessSize, 1, clampUnit(roughness), 0.0f);
}

NR_CPU_GPU inline float dielectricDirectionalAlbedo(
    const Textures* textures,
    const float roughness,
    const float cosine,
    const float normalReflectance)
{
#if defined(__CUDA_ARCH__)
    const uint16_t* hostValues = nullptr;
#else
    const uint16_t* hostValues = data::DielectricDirectionalAlbedo;
#endif
    return sample3d(textures ? textures->dielectricDirectional : 0, hostValues,
        DielectricCosineSize, DielectricRoughnessSize, DielectricF0Size,
        cosineCoordinate(cosine), clampUnit(roughness),
        dielectricF0Coordinate(normalReflectance));
}

NR_CPU_GPU inline float dielectricAverageAlbedo(
    const Textures* textures,
    const float roughness,
    const float normalReflectance)
{
#if defined(__CUDA_ARCH__)
    const uint16_t* hostValues = nullptr;
#else
    const uint16_t* hostValues = data::DielectricAverageAlbedo;
#endif
    return sample2d(textures ? textures->dielectricAverage : 0, hostValues,
        DielectricRoughnessSize, DielectricF0Size,
        clampUnit(roughness), dielectricF0Coordinate(normalReflectance));
}

NR_CPU_GPU inline float glassDirectionalAlbedo(
    const Textures* textures,
    const float roughness,
    const float cosine,
    const float relativeIor)
{
    if (fabsf(relativeIor - 1.0f) < 1.0e-4f)
        return 1.0f;
#if defined(__CUDA_ARCH__)
    const uint16_t* hostValues = nullptr;
#else
    const uint16_t* hostValues = data::GlassDirectionalAlbedo;
#endif
    return sample3d(textures ? textures->glassDirectional : 0, hostValues,
        GlassCosineSize, GlassRoughnessSize, GlassIorSize,
        cosineCoordinate(cosine), clampUnit(roughness),
        glassIorCoordinate(relativeIor));
}

NR_CPU_GPU inline float glassAverageAlbedo(
    const Textures* textures,
    const float roughness,
    const float relativeIor)
{
    if (fabsf(relativeIor - 1.0f) < 1.0e-4f)
        return 1.0f;
#if defined(__CUDA_ARCH__)
    const uint16_t* hostValues = nullptr;
#else
    const uint16_t* hostValues = data::GlassAverageAlbedo;
#endif
    return sample2d(textures ? textures->glassAverage : 0, hostValues,
        GlassRoughnessSize, GlassIorSize,
        clampUnit(roughness), glassIorCoordinate(relativeIor));
}

// Reference overloads keep direct LUT consumers concise. BSDF closures store
// a pointer to immutable scene LUTs, avoiding a 48-byte handle copy at every
// surface interaction.
NR_CPU_GPU inline float ggxDirectionalAlbedo(
    const Textures& textures, const float roughness, const float cosine)
{
    return ggxDirectionalAlbedo(&textures, roughness, cosine);
}

NR_CPU_GPU inline float ggxAverageAlbedo(
    const Textures& textures, const float roughness)
{
    return ggxAverageAlbedo(&textures, roughness);
}

NR_CPU_GPU inline float dielectricDirectionalAlbedo(
    const Textures& textures, const float roughness, const float cosine,
    const float normalReflectance)
{
    return dielectricDirectionalAlbedo(
        &textures, roughness, cosine, normalReflectance);
}

NR_CPU_GPU inline float dielectricAverageAlbedo(
    const Textures& textures, const float roughness,
    const float normalReflectance)
{
    return dielectricAverageAlbedo(
        &textures, roughness, normalReflectance);
}

NR_CPU_GPU inline float glassDirectionalAlbedo(
    const Textures& textures, const float roughness, const float cosine,
    const float relativeIor)
{
    return glassDirectionalAlbedo(&textures, roughness, cosine, relativeIor);
}

NR_CPU_GPU inline float glassAverageAlbedo(
    const Textures& textures, const float roughness, const float relativeIor)
{
    return glassAverageAlbedo(&textures, roughness, relativeIor);
}

// Kulla-Conty multi-scatter compensation: the energy a single-scatter GGX
// evaluation misses at high roughness, redistributed as an isotropic term.
// Shared by every microfacet lobe (Materials/Shading/Lobes/) that needs its own
// independent energy compensation shared by the active SVM microfacet lobes.
NR_CPU_GPU inline float multipleScatterFresnel(
    const float averageFresnel, const float averageAlbedo)
{
    const float fresnel = clampUnit(averageFresnel);
    const float albedo = clampUnit(averageAlbedo);
    return fresnel * fresnel * albedo
        / fmaxf(1.0f - fresnel * (1.0f - albedo), 1.0e-6f);
}

}
