#include "Samplers/RandomSampler.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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
