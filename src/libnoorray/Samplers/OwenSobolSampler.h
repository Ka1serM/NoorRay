#pragma once

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include <glm/vec2.hpp>

#include "Samplers/OwenSobolDirectionNumbers.h"
#include "Samplers/Sampler.h"

// Table-free Sobol dimensions with Burley's fast Owen scrambling.
class OwenSobolSampler : public IndexedSampler<OwenSobolSampler>
{
public:
    NR_CPU_GPU explicit OwenSobolSampler(const SampleKey key) : IndexedSampler(key) {}

    NR_CPU_GPU float evaluate1D(
        const uint32_t seed, const SampleDimension dimension) const
    {
        return scramble(sampleBits(key.index, static_cast<uint32_t>(dimension)), seed);
    }

    NR_CPU_GPU glm::vec2 evaluate2D(
        const uint32_t seedX, const uint32_t seedY,
        const SampleDimensionPair dimensions) const
    {
        return {
            scramble(sampleBits(key.index, static_cast<uint32_t>(dimensions.x)), seedX),
            scramble(sampleBits(key.index, static_cast<uint32_t>(dimensions.y)), seedY)};
    }

private:
    NR_CPU_GPU static uint32_t reverseBits(uint32_t value)
    {
#if defined(__CUDA_ARCH__)
        return __brev(value);
#else
        constexpr uint32_t AlternatingBits = 0x55555555u;
        constexpr uint32_t AlternatingPairs = 0x33333333u;
        constexpr uint32_t AlternatingNibbles = 0x0f0f0f0fu;
        constexpr uint32_t AlternatingBytes = 0x00ff00ffu;
        value = ((value & AlternatingBits) << 1u)
            | ((value >> 1u) & AlternatingBits);
        value = ((value & AlternatingPairs) << 2u)
            | ((value >> 2u) & AlternatingPairs);
        value = ((value & AlternatingNibbles) << 4u)
            | ((value >> 4u) & AlternatingNibbles);
        value = ((value & AlternatingBytes) << 8u)
            | ((value >> 8u) & AlternatingBytes);
        return (value << 16u) | (value >> 16u);
#endif
    }

    NR_CPU_GPU static uint32_t trailingZeroCount(const uint32_t value)
    {
#if defined(__CUDA_ARCH__)
        return static_cast<uint32_t>(__ffs(value) - 1);
#elif defined(_MSC_VER)
        unsigned long bit;
        _BitScanForward(&bit, value);
        return static_cast<uint32_t>(bit);
#else
        return static_cast<uint32_t>(__builtin_ctz(value));
#endif
    }

    NR_CPU_GPU static uint32_t directionNumber(
        const uint32_t dimension, const uint32_t bit)
    {
#if defined(__CUDA_ARCH__)
        return owen_sobol_detail::DeviceDirectionNumbers[dimension - 1u][bit];
#else
        return owen_sobol_detail::DirectionNumbers[dimension - 1u][bit];
#endif
    }

    NR_CPU_GPU static uint32_t sampleBits(
        const uint32_t sampleIndex, const uint32_t dimension)
    {
        if (dimension == 0)
            return reverseBits(sampleIndex);

        uint32_t value = 0;
        uint32_t index = sampleIndex;
        while (index != 0)
        {
            const uint32_t bit = trailingZeroCount(index);
            value ^= directionNumber(dimension, bit);
            index &= index - 1u;
        }
        return value;
    }

    NR_CPU_GPU static uint32_t fastOwenScramble(uint32_t value, const uint32_t seed)
    {
        // Fixed permutation constants from Burley's FastOwenScrambler.
        constexpr uint32_t InitialMultiplier = 0x3d20adeau;
        constexpr uint32_t FirstXorMultiplier = 0x05526c56u;
        constexpr uint32_t SecondXorMultiplier = 0x53a22864u;
        value = reverseBits(value);
        value ^= value * InitialMultiplier;
        value += seed;
        value *= (seed >> 16u) | 1u;
        value ^= value * FirstXorMultiplier;
        value ^= value * SecondXorMultiplier;
        return reverseBits(value);
    }

    NR_CPU_GPU static float scramble(const uint32_t bits, const uint32_t seed)
    {
        return unitFloatFromBits(fastOwenScramble(bits, seed));
    }
};
