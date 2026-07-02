#pragma once

#include <cmath>
#include <cstdint>

#include "CUDA/Annotations.h"

// Native NoorRay access to Adobe OpenPBR 1.1's opaque-dielectric energy
// compensation data. The source tables are vendored unchanged under external/
// and are Apache-2.0 licensed. Keeping the interpolation here avoids importing
// OpenPBR's shading-language vector ABI into NoorRay's GLM CUDA kernels.
namespace nr::openpbr
{
inline constexpr int EnergyTableSize = 32;
inline constexpr int EnergyTableSizeMinusOne = EnergyTableSize - 1;
inline constexpr float IorMax = 2.5f;

static constexpr uint16_t OpaqueDielectricEnergyComplement[
    EnergyTableSize * EnergyTableSize * EnergyTableSize] = {
#include <impl/data/openpbr_opaque_dielectric_energy_complement_data.h>
};

static constexpr uint16_t OpaqueDielectricAverageEnergyComplement[
    EnergyTableSize * EnergyTableSize] = {
#include <impl/data/openpbr_opaque_dielectric_avg_energy_complement_data.h>
};

static constexpr uint16_t IdealDielectricEnergyComplement[
    EnergyTableSize * EnergyTableSize * EnergyTableSize] = {
#include <impl/data/openpbr_ideal_dielectric_energy_complement_data.h>
};

static constexpr uint16_t IdealDielectricAverageEnergyComplement[
    EnergyTableSize * EnergyTableSize] = {
#include <impl/data/openpbr_ideal_dielectric_avg_energy_complement_data.h>
};

static constexpr uint16_t IdealDielectricReflectionRatio[
    EnergyTableSize * EnergyTableSize] = {
#include <impl/data/openpbr_ideal_dielectric_reflection_ratio_data.h>
};

#if defined(__CUDACC__)
static __device__ uint16_t OpaqueDielectricEnergyComplementDevice[
    EnergyTableSize * EnergyTableSize * EnergyTableSize] = {
#include <impl/data/openpbr_opaque_dielectric_energy_complement_data.h>
};
static __device__ uint16_t OpaqueDielectricAverageEnergyComplementDevice[
    EnergyTableSize * EnergyTableSize] = {
#include <impl/data/openpbr_opaque_dielectric_avg_energy_complement_data.h>
};
static __device__ uint16_t IdealDielectricEnergyComplementDevice[
    EnergyTableSize * EnergyTableSize * EnergyTableSize] = {
#include <impl/data/openpbr_ideal_dielectric_energy_complement_data.h>
};
static __device__ uint16_t IdealDielectricAverageEnergyComplementDevice[
    EnergyTableSize * EnergyTableSize] = {
#include <impl/data/openpbr_ideal_dielectric_avg_energy_complement_data.h>
};
static __device__ uint16_t IdealDielectricReflectionRatioDevice[
    EnergyTableSize * EnergyTableSize] = {
#include <impl/data/openpbr_ideal_dielectric_reflection_ratio_data.h>
};
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

NR_CPU_GPU inline float opaqueEnergyAt(const int ior, const int alpha, const int cosine)
{
#if defined(__CUDA_ARCH__)
    return decode(OpaqueDielectricEnergyComplementDevice[
        ior * EnergyTableSize * EnergyTableSize + alpha * EnergyTableSize + cosine]);
#else
    return decode(OpaqueDielectricEnergyComplement[
        ior * EnergyTableSize * EnergyTableSize + alpha * EnergyTableSize + cosine]);
#endif
}

NR_CPU_GPU inline float opaqueAverageEnergyAt(const int ior, const int alpha)
{
#if defined(__CUDA_ARCH__)
    return decode(OpaqueDielectricAverageEnergyComplementDevice[ior * EnergyTableSize + alpha]);
#else
    return decode(OpaqueDielectricAverageEnergyComplement[ior * EnergyTableSize + alpha]);
#endif
}

NR_CPU_GPU inline float idealEnergyAt(const int ior, const int alpha, const int cosine)
{
#if defined(__CUDA_ARCH__)
    return decode(IdealDielectricEnergyComplementDevice[
        ior * EnergyTableSize * EnergyTableSize + alpha * EnergyTableSize + cosine]);
#else
    return decode(IdealDielectricEnergyComplement[
        ior * EnergyTableSize * EnergyTableSize + alpha * EnergyTableSize + cosine]);
#endif
}

NR_CPU_GPU inline float idealAverageEnergyAt(const int ior, const int alpha)
{
#if defined(__CUDA_ARCH__)
    return decode(IdealDielectricAverageEnergyComplementDevice[ior * EnergyTableSize + alpha]);
#else
    return decode(IdealDielectricAverageEnergyComplement[ior * EnergyTableSize + alpha]);
#endif
}

NR_CPU_GPU inline float idealReflectionRatioAt(const int ior, const int alpha)
{
#if defined(__CUDA_ARCH__)
    return decode(IdealDielectricReflectionRatioDevice[ior * EnergyTableSize + alpha]);
#else
    return decode(IdealDielectricReflectionRatio[ior * EnergyTableSize + alpha]);
#endif
}

NR_CPU_GPU inline float opaqueDielectricEnergyComplement(
    const float ior, const float alpha, const float cosTheta)
{
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
}

NR_CPU_GPU inline float opaqueDielectricAverageEnergyComplement(
    const float ior, const float alpha)
{
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
}

NR_CPU_GPU inline float idealDielectricEnergyComplement(
    const float ior, const float alpha, const float cosTheta)
{
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
}

NR_CPU_GPU inline float idealDielectricAverageEnergyComplement(
    const float ior, const float alpha)
{
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
}

NR_CPU_GPU inline float idealDielectricReflectionRatio(
    const float ior, const float alpha)
{
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
}
}
