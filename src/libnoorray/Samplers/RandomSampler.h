#pragma once

#include <cstdint>

#include "CUDA/Annotations.h"

using RandomState = uint64_t;

NR_CPU_GPU inline uint32_t pcgHash(uint32_t value)
{
    value = value * 747796405u + 2891336453u;
    const uint32_t shift = (value >> 28u) + 4u;
    return ((value >> shift) ^ value) * 277803737u ^ value;
}

NR_CPU_GPU inline float randomFloat(uint32_t& state)
{
    state = pcgHash(state);
    return static_cast<float>(state >> 8u) * (1.0f / 16777216.0f);
}

NR_CPU_GPU inline uint64_t splitMix64(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

NR_CPU_GPU inline RandomState seedRandom(const uint64_t sequence)
{
    return splitMix64(sequence) | 1ull;
}

// PCG-XSH-RR: unlike repeatedly hashing a 32-bit word, this is a stateful
// generator with a 2^64 period and a well-tested output permutation.
NR_CPU_GPU inline uint32_t randomUint(RandomState& state)
{
    const uint64_t oldState = state;
    state = oldState * 6364136223846793005ull + 1442695040888963407ull;
    const uint32_t xorshifted = static_cast<uint32_t>(((oldState >> 18u) ^ oldState) >> 27u);
    const uint32_t rotation = static_cast<uint32_t>(oldState >> 59u);
    return (xorshifted >> rotation) | (xorshifted << ((-rotation) & 31u));
}

NR_CPU_GPU inline float randomFloat(RandomState& state)
{
    return static_cast<float>(randomUint(state) >> 8u) * (1.0f / 16777216.0f);
}
