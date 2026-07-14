#pragma once

#include <cstdint>

#include <glm/vec2.hpp>

#include "Samplers/Sampler.h"

// Table-free first two Sobol dimensions with Burley's fast Owen scrambling.
// One raw point can be independently scrambled for padded 1D/2D dimensions.
class OwenSobolSampler : public IndexedSampler<OwenSobolSampler>
{
public:
    NR_CPU_GPU explicit OwenSobolSampler(const SampleKey key)
        : IndexedSampler(key), bits(sampleBits(key.index))
    {
    }

    NR_CPU_GPU float evaluate1D(
        const uint32_t seed, const SampleDimension dimension) const
    {
        const uint32_t component = static_cast<uint32_t>(dimension) & 1u;
        return scramble(component == 0 ? bits.x : bits.y, seed);
    }

    NR_CPU_GPU glm::vec2 evaluate2D(
        const uint32_t seedX, const uint32_t seedY, SampleDimensionPair) const
    {
        return {scramble(bits.x, seedX), scramble(bits.y, seedY)};
    }

private:
    glm::uvec2 bits;

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

    NR_CPU_GPU static glm::uvec2 sampleBits(uint32_t index)
    {
        const uint32_t x = reverseBits(index);
        uint32_t y = 0;
        uint32_t direction = uint32_t{1} << 31u;
        while (index != 0)
        {
            if ((index & 1u) != 0)
                y ^= direction;
            index >>= 1u;
            direction ^= direction >> 1u;
        }
        return {x, y};
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
