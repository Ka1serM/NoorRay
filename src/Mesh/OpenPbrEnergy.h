#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "CUDA/Annotations.h"
#ifndef NR_GPU_CODE
#include "CUDA/Unique/Texture.h"
#endif

// Native NoorRay access to Adobe OpenPBR 1.1's opaque-dielectric energy
// compensation data. The source tables are vendored unchanged under
// Mesh/OpenPbrEnergyTables/ and are Apache-2.0 licensed. Keeping the
// interpolation here avoids importing OpenPBR's shading-language vector ABI
// into NoorRay's GLM CUDA kernels.
//
// On the GPU, lookups are done with hardware-filtered CUDA textures (linear
// filtering does the tri-/bilinear blend for free), which is both faster and
// simpler than the manual multi-tap lerp below. That manual path is kept as
// the CPU fallback so host-side unit tests can exercise the exact same energy
// data without a CUDA context.
namespace nr::openpbr
{
inline constexpr int EnergyTableSize = 32;
inline constexpr int EnergyTableSizeMinusOne = EnergyTableSize - 1;
inline constexpr float IorMax = 2.5f;

static constexpr uint16_t OpaqueDielectricEnergyComplement[
    EnergyTableSize * EnergyTableSize * EnergyTableSize] = {
#include "Mesh/OpenPbrEnergyTables/openpbr_opaque_dielectric_energy_complement_data.h"
};

static constexpr uint16_t OpaqueDielectricAverageEnergyComplement[
    EnergyTableSize * EnergyTableSize] = {
#include "Mesh/OpenPbrEnergyTables/openpbr_opaque_dielectric_avg_energy_complement_data.h"
};

static constexpr uint16_t IdealDielectricEnergyComplement[
    EnergyTableSize * EnergyTableSize * EnergyTableSize] = {
#include "Mesh/OpenPbrEnergyTables/openpbr_ideal_dielectric_energy_complement_data.h"
};

static constexpr uint16_t IdealDielectricAverageEnergyComplement[
    EnergyTableSize * EnergyTableSize] = {
#include "Mesh/OpenPbrEnergyTables/openpbr_ideal_dielectric_avg_energy_complement_data.h"
};

static constexpr uint16_t IdealDielectricReflectionRatio[
    EnergyTableSize * EnergyTableSize] = {
#include "Mesh/OpenPbrEnergyTables/openpbr_ideal_dielectric_reflection_ratio_data.h"
};

// Device-visible texture handles, threaded through GpuSceneData like the
// other global LUTs (spectrumTable*, cieX/Y/Z). Hardware linear filtering
// with cudaReadModeNormalizedFloat replaces decode()+lerp() entirely.
struct EnergyLutTextures
{
    cudaTextureObject_t opaqueEnergy{};
    cudaTextureObject_t opaqueAverageEnergy{};
    cudaTextureObject_t idealEnergy{};
    cudaTextureObject_t idealAverageEnergy{};
    cudaTextureObject_t idealReflectionRatio{};
};

#ifndef NR_GPU_CODE
// Host-owned backing arrays and texture objects. The raw texture handles are
// copied into EnergyLutTextures for device code.
struct EnergyLutStorage
{
    nr::cuda::UniqueTexture opaqueEnergy;
    nr::cuda::UniqueTexture opaqueAverageEnergy;
    nr::cuda::UniqueTexture idealEnergy;
    nr::cuda::UniqueTexture idealAverageEnergy;
    nr::cuda::UniqueTexture idealReflectionRatio;

    void reset() noexcept { *this = {}; }
};

// Uploads the tables above into 3D/2D CUDA arrays and creates linear-filtered
// texture objects over them. Call once at startup.
void uploadEnergyLuts(EnergyLutStorage& storage, EnergyLutTextures& textures, cudaStream_t stream);
EnergyLutTextures getEnergyLutTextures(const EnergyLutStorage& storage) noexcept;

#endif

NR_CPU_GPU inline float decode(const uint16_t value)
{
    return static_cast<float>(value) * (1.0f / 65535.0f);
}

NR_CPU_GPU inline float iorIndex(const float ior)
{
    constexpr float inverseRange = 1.0f / (IorMax - 1.0f);
    constexpr int half = EnergyTableSize / 2;
    if (ior < 1.0f)
        return static_cast<float>(half - 1)
            - (1.0f / ior - 1.0f) * inverseRange * static_cast<float>(half - 1);
    return static_cast<float>(half)
        + (ior - 1.0f) * inverseRange * static_cast<float>(half - 1);
}

NR_CPU_GPU inline float clampedIndex(const float value)
{
    return fminf(fmaxf(value, 0.0f), static_cast<float>(EnergyTableSizeMinusOne));
}

NR_CPU_GPU inline float lerp(const float a, const float b, const float t)
{
    return a + (b - a) * t;
}

// CPU-only fallback path below: manual multi-tap lerp over the host arrays,
// bit-identical to the pre-texture implementation.

inline float opaqueEnergyAt(const int ior, const int alpha, const int cosine)
{
    return decode(OpaqueDielectricEnergyComplement[
        ior * EnergyTableSize * EnergyTableSize + alpha * EnergyTableSize + cosine]);
}

inline float opaqueAverageEnergyAt(const int ior, const int alpha)
{
    return decode(OpaqueDielectricAverageEnergyComplement[ior * EnergyTableSize + alpha]);
}

inline float idealEnergyAt(const int ior, const int alpha, const int cosine)
{
    return decode(IdealDielectricEnergyComplement[
        ior * EnergyTableSize * EnergyTableSize + alpha * EnergyTableSize + cosine]);
}

inline float idealAverageEnergyAt(const int ior, const int alpha)
{
    return decode(IdealDielectricAverageEnergyComplement[ior * EnergyTableSize + alpha]);
}

inline float idealReflectionRatioAt(const int ior, const int alpha)
{
    return decode(IdealDielectricReflectionRatio[ior * EnergyTableSize + alpha]);
}

NR_CPU_GPU inline float opaqueDielectricEnergyComplement(
    const EnergyLutTextures& luts, const float ior, const float alpha, const float cosTheta)
{
#if defined(__CUDA_ARCH__)
    // Host array is ior-major (ior*N*N + alpha*N + cosine): cosine varies
    // fastest, ior slowest. CUDA's 3D array x/y/z map fastest→slowest, so the
    // tex3D argument order is (cosine, alpha, ior), not (ior, alpha, cosine).
    const float iorCoord = clampedIndex(iorIndex(ior)) + 0.5f;
    const float alphaCoord = clampedIndex(sqrtf(fmaxf(alpha, 0.0f)) * EnergyTableSizeMinusOne) + 0.5f;
    const float cosineCoord = clampedIndex(cosTheta * EnergyTableSizeMinusOne) + 0.5f;
    return tex3D<float>(luts.opaqueEnergy, cosineCoord, alphaCoord, iorCoord);
#else
    (void)luts;
    const float x = clampedIndex(iorIndex(ior));
    const float y = clampedIndex(sqrtf(fmaxf(alpha, 0.0f)) * EnergyTableSizeMinusOne);
    const float z = clampedIndex(cosTheta * EnergyTableSizeMinusOne);
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int z0 = static_cast<int>(floorf(z));
    const int x1 = x0 < EnergyTableSizeMinusOne ? x0 + 1 : x0;
    const int y1 = y0 < EnergyTableSizeMinusOne ? y0 + 1 : y0;
    const int z1 = z0 < EnergyTableSizeMinusOne ? z0 + 1 : z0;
    const float tx = x - x0;
    const float ty = y - y0;
    const float tz = z - z0;
    const float v00 = lerp(opaqueEnergyAt(x0, y0, z0), opaqueEnergyAt(x1, y0, z0), tx);
    const float v01 = lerp(opaqueEnergyAt(x0, y0, z1), opaqueEnergyAt(x1, y0, z1), tx);
    const float v10 = lerp(opaqueEnergyAt(x0, y1, z0), opaqueEnergyAt(x1, y1, z0), tx);
    const float v11 = lerp(opaqueEnergyAt(x0, y1, z1), opaqueEnergyAt(x1, y1, z1), tx);
    return lerp(lerp(v00, v10, ty), lerp(v01, v11, ty), tz);
#endif
}

NR_CPU_GPU inline float opaqueDielectricAverageEnergyComplement(
    const EnergyLutTextures& luts, const float ior, const float alpha)
{
#if defined(__CUDA_ARCH__)
    // Host array is ior-major (ior*N + alpha): alpha varies fastest, ior
    // slowest. CUDA's 2D array x/y map fastest→slowest, so tex2D takes
    // (alpha, ior), not (ior, alpha).
    const float iorCoord = clampedIndex(iorIndex(ior)) + 0.5f;
    const float alphaCoord = clampedIndex(sqrtf(fmaxf(alpha, 0.0f)) * EnergyTableSizeMinusOne) + 0.5f;
    return tex2D<float>(luts.opaqueAverageEnergy, alphaCoord, iorCoord);
#else
    (void)luts;
    const float x = clampedIndex(iorIndex(ior));
    const float y = clampedIndex(sqrtf(fmaxf(alpha, 0.0f)) * EnergyTableSizeMinusOne);
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = x0 < EnergyTableSizeMinusOne ? x0 + 1 : x0;
    const int y1 = y0 < EnergyTableSizeMinusOne ? y0 + 1 : y0;
    const float tx = x - x0;
    const float ty = y - y0;
    return lerp(lerp(opaqueAverageEnergyAt(x0, y0), opaqueAverageEnergyAt(x1, y0), tx),
        lerp(opaqueAverageEnergyAt(x0, y1), opaqueAverageEnergyAt(x1, y1), tx), ty);
#endif
}

NR_CPU_GPU inline float idealDielectricEnergyComplement(
    const EnergyLutTextures& luts, const float ior, const float alpha, const float cosTheta)
{
#if defined(__CUDA_ARCH__)
    // Host array is ior-major (ior*N*N + alpha*N + cosine): cosine varies
    // fastest, ior slowest. CUDA's 3D array x/y/z map fastest→slowest, so the
    // tex3D argument order is (cosine, alpha, ior), not (ior, alpha, cosine).
    const float iorCoord = clampedIndex(iorIndex(ior)) + 0.5f;
    const float alphaCoord = clampedIndex(sqrtf(fmaxf(alpha, 0.0f)) * EnergyTableSizeMinusOne) + 0.5f;
    const float cosineCoord = clampedIndex(cosTheta * EnergyTableSizeMinusOne) + 0.5f;
    return tex3D<float>(luts.idealEnergy, cosineCoord, alphaCoord, iorCoord);
#else
    (void)luts;
    const float x = clampedIndex(iorIndex(ior));
    const float y = clampedIndex(sqrtf(fmaxf(alpha, 0.0f)) * EnergyTableSizeMinusOne);
    const float z = clampedIndex(cosTheta * EnergyTableSizeMinusOne);
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int z0 = static_cast<int>(floorf(z));
    const int x1 = x0 < EnergyTableSizeMinusOne ? x0 + 1 : x0;
    const int y1 = y0 < EnergyTableSizeMinusOne ? y0 + 1 : y0;
    const int z1 = z0 < EnergyTableSizeMinusOne ? z0 + 1 : z0;
    const float tx = x - x0;
    const float ty = y - y0;
    const float tz = z - z0;
    const float v00 = lerp(idealEnergyAt(x0, y0, z0), idealEnergyAt(x1, y0, z0), tx);
    const float v01 = lerp(idealEnergyAt(x0, y0, z1), idealEnergyAt(x1, y0, z1), tx);
    const float v10 = lerp(idealEnergyAt(x0, y1, z0), idealEnergyAt(x1, y1, z0), tx);
    const float v11 = lerp(idealEnergyAt(x0, y1, z1), idealEnergyAt(x1, y1, z1), tx);
    return lerp(lerp(v00, v10, ty), lerp(v01, v11, ty), tz);
#endif
}

NR_CPU_GPU inline float idealDielectricAverageEnergyComplement(
    const EnergyLutTextures& luts, const float ior, const float alpha)
{
#if defined(__CUDA_ARCH__)
    // Host array is ior-major (ior*N + alpha): alpha varies fastest, ior
    // slowest. CUDA's 2D array x/y map fastest→slowest, so tex2D takes
    // (alpha, ior), not (ior, alpha).
    const float iorCoord = clampedIndex(iorIndex(ior)) + 0.5f;
    const float alphaCoord = clampedIndex(sqrtf(fmaxf(alpha, 0.0f)) * EnergyTableSizeMinusOne) + 0.5f;
    return tex2D<float>(luts.idealAverageEnergy, alphaCoord, iorCoord);
#else
    (void)luts;
    const float x = clampedIndex(iorIndex(ior));
    const float y = clampedIndex(sqrtf(fmaxf(alpha, 0.0f)) * EnergyTableSizeMinusOne);
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = x0 < EnergyTableSizeMinusOne ? x0 + 1 : x0;
    const int y1 = y0 < EnergyTableSizeMinusOne ? y0 + 1 : y0;
    const float tx = x - x0;
    const float ty = y - y0;
    return lerp(lerp(idealAverageEnergyAt(x0, y0), idealAverageEnergyAt(x1, y0), tx),
        lerp(idealAverageEnergyAt(x0, y1), idealAverageEnergyAt(x1, y1), tx), ty);
#endif
}

NR_CPU_GPU inline float idealDielectricReflectionRatio(
    const EnergyLutTextures& luts, const float ior, const float alpha)
{
#if defined(__CUDA_ARCH__)
    // Host array is ior-major (ior*N + alpha): alpha varies fastest, ior
    // slowest. CUDA's 2D array x/y map fastest→slowest, so tex2D takes
    // (alpha, ior), not (ior, alpha).
    const float iorCoord = clampedIndex(iorIndex(ior)) + 0.5f;
    const float alphaCoord = clampedIndex(sqrtf(fmaxf(alpha, 0.0f)) * EnergyTableSizeMinusOne) + 0.5f;
    return tex2D<float>(luts.idealReflectionRatio, alphaCoord, iorCoord);
#else
    (void)luts;
    const float x = clampedIndex(iorIndex(ior));
    const float y = clampedIndex(sqrtf(fmaxf(alpha, 0.0f)) * EnergyTableSizeMinusOne);
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = x0 < EnergyTableSizeMinusOne ? x0 + 1 : x0;
    const int y1 = y0 < EnergyTableSizeMinusOne ? y0 + 1 : y0;
    const float tx = x - x0;
    const float ty = y - y0;
    return lerp(lerp(idealReflectionRatioAt(x0, y0), idealReflectionRatioAt(x1, y0), tx),
        lerp(idealReflectionRatioAt(x0, y1), idealReflectionRatioAt(x1, y1), tx), ty);
#endif
}
}
