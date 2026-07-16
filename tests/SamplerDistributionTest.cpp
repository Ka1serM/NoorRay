#include "Samplers/RandomSampler.h"
#include "Samplers/OwenSobolSampler.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>

class SamplerDistributionTest {};

TEST_CASE_METHOD(SamplerDistributionTest,
    "randomFloat is deterministic, bounded, and uniform", "[sampler]")
{
    RandomState a = seedRandom(42u);
    RandomState b = seedRandom(42u);
    RandomState c = seedRandom(43u);
    double sum = 0.0;
    bool differs = false;
    constexpr int sampleCount = 100000;
    for (int i = 0; i < sampleCount; ++i) {
        const float x = randomFloat(a);
        const float y = randomFloat(b);
        const float z = randomFloat(c);
        REQUIRE(x == y);
        REQUIRE(x >= 0.0f);
        REQUIRE(x < 1.0f);
        differs |= x != z;
        sum += x;
    }

    CHECK(differs);
    CHECK(sum / sampleCount == Catch::Approx(0.5).margin(0.01));
}

TEST_CASE_METHOD(SamplerDistributionTest,
    "Owen Sobol dimensions retain joint stratification", "[sampler]")
{
    constexpr uint32_t strataPerDimension = 8;
    constexpr uint32_t sampleCount = strataPerDimension * strataPerDimension;
    std::array<bool, sampleCount> wavelengthLensCells{};

    for (uint32_t sample = 0; sample < sampleCount; ++sample) {
        const OwenSobolSampler sampler({sample, 12345u});
        const float wavelength = sampler.sample1D(SampleDimension::Wavelength);
        const float lensX = sampler.sample2D(LensSampleDimensions).x;
        const uint32_t x = static_cast<uint32_t>(wavelength * strataPerDimension);
        const uint32_t y = static_cast<uint32_t>(lensX * strataPerDimension);
        wavelengthLensCells[y * strataPerDimension + x] = true;
    }

    uint32_t occupiedCells = 0;
    for (const bool occupied : wavelengthLensCells)
        occupiedCells += occupied ? 1u : 0u;

    // The old padded implementation covered only 8 cells because wavelength
    // and lens X were permutations of the same one-dimensional coordinate.
    CHECK(occupiedCells >= sampleCount / 2u);
}

TEST_CASE_METHOD(SamplerDistributionTest,
    "Owen Sobol sequence remains bit exact", "[sampler]")
{
    constexpr std::array<uint32_t, 6> SampleIndices{
        0u, 1u, 2u, 17u, 123456789u, 0xffffffffu};
    constexpr std::size_t DimensionCount =
        static_cast<std::size_t>(SampleDimension::Count);
    constexpr std::array<std::array<uint32_t, DimensionCount>, SampleIndices.size()> Expected{{
        {0x3eadb911u, 0x3e55e9c2u, 0x3f6b4f4eu, 0x3edbc4bdu, 0x3f7cb23eu, 0x3f5136f8u},
        {0x3f2c0abeu, 0x3f68514eu, 0x3edd94c3u, 0x3f783886u, 0x3bb7d4c0u, 0x3de0e4b4u},
        {0x3b63c180u, 0x3f3b604cu, 0x3caaa330u, 0x3f075f62u, 0x3f2ff5ceu, 0x3f0bee38u},
        {0x3f27b3e6u, 0x3e656ea2u, 0x3f3345b6u, 0x3e6987c2u, 0x3f092f82u, 0x3f05c7bau},
        {0x3f1e6550u, 0x3e8c049fu, 0x3e687246u, 0x3f03b8d2u, 0x3f03f238u, 0x3e1f3a1au},
        {0x3f6362a6u, 0x3e55e9c2u, 0x3f07d24cu, 0x3ebec017u, 0x3f3ea478u, 0x3edf951fu},
    }};

    for (uint32_t sample = 0; sample < SampleIndices.size(); ++sample) {
        const OwenSobolSampler sampler({SampleIndices[sample], 12345u});
        for (uint32_t dimension = 0;
             dimension < static_cast<uint32_t>(SampleDimension::Count); ++dimension) {
            const float value = sampler.sample1D(static_cast<SampleDimension>(dimension));
            CHECK(std::bit_cast<uint32_t>(value) == Expected[sample][dimension]);
        }
    }
}
