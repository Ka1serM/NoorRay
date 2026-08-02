#pragma once

#include <cstdint>

#include "Backend/CUDA/Annotations.h"

using RandomState = uint64_t;

enum class RandomStream : uint32_t
{
    Opacity,
    Bsdf,
    Light,
    Shadow,
    Roulette,
};

NR_CPU_GPU inline float unitFloatFromBits(const uint32_t bits)
{
    constexpr uint32_t DiscardedLowBits = 8u;
    constexpr float BinCenter = 0.5f;
    constexpr float InverseBinCount = 1.0f / 16777216.0f;
    return (static_cast<float>(bits >> DiscardedLowBits) + BinCenter) * InverseBinCount;
}

NR_CPU_GPU inline uint32_t pcgHash(uint32_t value)
{
    // PCG RXS-M-XS constants from O'Neill's PCG reference implementation.
    constexpr uint32_t Multiplier = 747796405u;
    constexpr uint32_t Increment = 2891336453u;
    constexpr uint32_t OutputMultiplier = 277803737u;
    constexpr uint32_t VariableShiftBase = 4u;
    constexpr uint32_t VariableShiftSource = 28u;
    value = value * Multiplier + Increment;
    const uint32_t shift = (value >> VariableShiftSource) + VariableShiftBase;
    return ((value >> shift) ^ value) * OutputMultiplier ^ value;
}

NR_CPU_GPU inline uint32_t hashCombine32(const uint32_t first, const uint32_t second)
{
    return pcgHash(first ^ pcgHash(second));
}

NR_CPU_GPU inline uint32_t sampleDimensionSeed(
    const uint32_t scramble, const uint32_t dimension)
{
    return hashCombine32(scramble, dimension);
}

NR_CPU_GPU inline float hashFloat(const uint32_t value)
{
    return unitFloatFromBits(pcgHash(value));
}

NR_CPU_GPU inline float randomFloat(uint32_t& state)
{
    state = pcgHash(state);
    return unitFloatFromBits(state);
}

NR_CPU_GPU inline uint64_t splitMix64(uint64_t value)
{
    // SplitMix64 constants from Steele, Lea, and Flood's reference mixer.
    constexpr uint64_t WeylIncrement = 0x9e3779b97f4a7c15ull;
    constexpr uint64_t FirstMultiplier = 0xbf58476d1ce4e5b9ull;
    constexpr uint64_t SecondMultiplier = 0x94d049bb133111ebull;
    constexpr uint32_t FirstShift = 30u;
    constexpr uint32_t SecondShift = 27u;
    constexpr uint32_t FinalShift = 31u;
    value += WeylIncrement;
    value = (value ^ (value >> FirstShift)) * FirstMultiplier;
    value = (value ^ (value >> SecondShift)) * SecondMultiplier;
    return value ^ (value >> FinalShift);
}

NR_CPU_GPU inline RandomState seedRandom(const uint64_t sequence)
{
    constexpr uint64_t RequiredOddBit = 1ull;
    return splitMix64(sequence) | RequiredOddBit;
}

NR_CPU_GPU inline RandomState advanceRandomSequence(const RandomState state)
{
    return splitMix64(state);
}

NR_CPU_GPU inline RandomState forkRandom(
    const RandomState base, const RandomStream stream)
{
    return seedRandom(base ^ splitMix64(static_cast<uint64_t>(stream)));
}

// PCG-XSH-RR: unlike repeatedly hashing a 32-bit word, this is a stateful
// generator with a 2^64 period and a well-tested output permutation.
NR_CPU_GPU inline uint32_t randomUint(RandomState& state)
{
    constexpr uint64_t StateMultiplier = 6364136223846793005ull;
    constexpr uint64_t StateIncrement = 1442695040888963407ull;
    constexpr uint32_t XorShift = 18u;
    constexpr uint32_t OutputShift = 27u;
    constexpr uint32_t RotationShift = 59u;
    const uint64_t oldState = state;
    state = oldState * StateMultiplier + StateIncrement;
    const uint32_t xorshifted = static_cast<uint32_t>(
        ((oldState >> XorShift) ^ oldState) >> OutputShift);
    const uint32_t rotation = static_cast<uint32_t>(oldState >> RotationShift);
    return (xorshifted >> rotation) | (xorshifted << ((-rotation) & 31u));
}

NR_CPU_GPU inline float randomFloat(RandomState& state)
{
    return unitFloatFromBits(randomUint(state));
}
