#include "BsdfTestFixture.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

class DielectricBsdfTest : public BsdfTestFixture {};

TEST_CASE_METHOD(DielectricBsdfTest,
    "microfacet dielectric sampling produces valid reflection and transmission",
    "[bsdf][dielectric]")
{
    constexpr int sampleCount = 100000;
    RandomState rng = seedRandom(0x5eedu);
    int reflections = 0;
    int rejected = 0;
    for (int i = 0; i < sampleCount; ++i) {
        const BsdfSample sample = Bsdf::sampleDielectric(
            view, normal, normal, 0.045f, 1.5f, rng);
        REQUIRE(std::isfinite(sample.pdf));
        if (sample.pdf == 0.0f) {
            ++rejected;
            CHECK(sample.weight.maxComponent() == 0.0f);
            continue;
        }
        REQUIRE(sample.pdf > 0.0f);
        REQUIRE(std::isfinite(sample.weight[0]));
        REQUIRE(sample.weight[0] >= 0.0f);
        CHECK(glm::length(sample.direction) == Catch::Approx(1.0f).margin(2e-5f));
        if (sample.event == BsdfEvent::Specular) {
            ++reflections;
            CHECK(sample.direction.z > 0.0f);
        } else {
            CHECK(sample.event == BsdfEvent::Transmission);
            CHECK(sample.direction.z < 0.0f);
            CHECK(sample.eta == Catch::Approx(1.5f));
        }
    }

    const float reflectionRate = static_cast<float>(reflections) / sampleCount;
    CHECK(reflectionRate == Catch::Approx(0.04f).margin(0.003f));
    CHECK(rejected < sampleCount / 1000);
}
