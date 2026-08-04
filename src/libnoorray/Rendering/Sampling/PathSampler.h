#pragma once

#include <cstdint>

#include "Rendering/Sampling/OwenSobolSampler.h"

// A mutable view of one logical path-sampling block. Each four draws use the
// first four Sobol dimensions jointly; later draws advance to an independently
// Owen-scrambled block. This mirrors Cycles' padding strategy and, unlike a
// stateful PRNG, leaves independent path decisions unaffected by branching.
struct PathSampleStream
{
    uint32_t sampleIndex{};
    uint32_t pixelScramble{};
    uint32_t block{};
    uint32_t draw{};
};

NR_CPU_GPU inline uint32_t randomUint(PathSampleStream& stream)
{
    constexpr uint32_t DimensionsPerBlock = 4u;
    const uint32_t block = stream.block + stream.draw / DimensionsPerBlock;
    const uint32_t dimension = stream.draw % DimensionsPerBlock;
    ++stream.draw;

    const uint32_t blockSeed = hashCombine32(stream.pixelScramble, block);
    return OwenSobolSampler::sampleUint(
        stream.sampleIndex, dimension,
        hashCombine32(blockSeed, 0xd1b54a35u + dimension));
}

NR_CPU_GPU inline float randomFloat(PathSampleStream& stream)
{
    return unitFloatFromBits(randomUint(stream));
}
