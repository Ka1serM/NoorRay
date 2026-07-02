#include "BsdfFixture.h"

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
    const BsdfMaterialParameters material = makeMaterial(
        1.0f, 0.0f, 0.5f, 0.045f, 1.0f, 1.5f);
    const SampledWavelengths wl = wavelengths();
    int reflections = 0;
    for (int i = 0; i < sampleCount; ++i) {
        const BsdfSample sample = nr::bsdf::sample(
            material, normal, normal, view, rng, wl);
        REQUIRE(std::isfinite(sample.pdf));
        REQUIRE(sample.pdf > 0.0f);
        REQUIRE(std::isfinite(sample.weight[0]));
        REQUIRE(sample.weight[0] >= 0.0f);
        CHECK(glm::length(sample.direction) == Catch::Approx(1.0f).margin(2e-5f));
        if (sample.event != BsdfEvent::Transmission) {
            ++reflections;
            CHECK(sample.direction.z > 0.0f);
        } else {
            CHECK(sample.direction.z < 0.0f);
            for (int channel = 1; channel < NrSpectrumSamples; ++channel)
                CHECK(sample.weight[channel] > 0.0f);
            CHECK_FALSE(sample.terminateSecondaryWavelengths);
        }
    }

    const float reflectionRate = static_cast<float>(reflections) / sampleCount;
    CHECK(reflectionRate == Catch::Approx(0.04f).margin(0.003f));
}

TEST_CASE_METHOD(DielectricBsdfTest,
    "hero-wavelength transmission terminates secondary spectral samples",
    "[bsdf][dielectric][spectral]")
{
    const BsdfMaterialParameters material = makeMaterial(
        1.0f, 0.0f, 0.5f, 0.045f, 1.0f, 1.5f);
    BsdfMaterialParameters dispersiveMaterial = material;
    dispersiveMaterial.sellmeier = fitSellmeierFromFraunhofer(
        glm::vec3(1.49f, 1.50f, 1.52f));
    SampledWavelengths wl = wavelengths();
    BsdfSample sample{};
    for (uint64_t attempt = 0; attempt < 64; ++attempt) {
        RandomState rng = seedRandom(attempt);
        sample = nr::bsdf::sample(dispersiveMaterial, normal, normal, view, rng, wl);
        if (sample.event == BsdfEvent::Transmission)
            break;
    }

    REQUIRE(sample.event == BsdfEvent::Transmission);
    REQUIRE(sample.terminateSecondaryWavelengths);
    REQUIRE(sample.weight[0] > 0.0f);
    for (int i = 1; i < NrSpectrumSamples; ++i)
        CHECK(sample.weight[i] == 0.0f);

    const float originalHeroPdf = wl.pdf[0];
    wl.terminateSecondary();
    CHECK(wl.pdf[0] == Catch::Approx(originalHeroPdf / NrSpectrumSamples));
    for (int i = 1; i < NrSpectrumSamples; ++i)
        CHECK(wl.pdf[i] == 0.0f);
}
