#include "Samplers/RandomSampler.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>

class SamplerSequenceTest {};

TEST_CASE_METHOD(SamplerSequenceTest, "randomUint sequence remains stable", "[sampler]")
{
    RandomState state = 0x123456789abcdef1ull;
    constexpr std::array<uint32_t, 6> expected{
        1369610156u, 2360205090u, 1914006763u,
        4290528257u, 4267313707u, 292903945u};

    for (const uint32_t value : expected)
        CHECK(randomUint(state) == value);
}
