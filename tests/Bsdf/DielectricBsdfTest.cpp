#include "BsdfTestFixture.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>

class DielectricBsdfTest : public BsdfTestFixture {};

TEST_CASE_METHOD(DielectricBsdfTest,
    "microfacet dielectric sampling produces valid reflection and transmission",
    "[bsdf][dielectric]")
{
    constexpr int sampleCount = 100000;
    std::mt19937 rng(0x5eedu);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const OpenPbrMaterialParameters material = makeMaterial(
        1.0f, 0.0f, 0.5f, 0.045f, 1.0f, 1.5f);
    const SampledWavelengths wl = wavelengths();
    int reflections = 0;
    for (int i = 0; i < sampleCount; ++i) {
        const BsdfSample sample = nr::openpbr::sample(
            material, normal, normal, view,
            glm::vec3(uniform(rng), uniform(rng), uniform(rng)),
            wl, unitThroughput);
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
    const OpenPbrMaterialParameters material = makeMaterial(
        1.0f, 0.0f, 0.5f, 0.045f, 1.0f, 1.5f);
    OpenPbrMaterialParameters dispersiveMaterial = material;
    dispersiveMaterial.sellmeier = fitSellmeierFromFraunhofer(
        glm::vec3(1.49f, 1.50f, 1.52f));
    SampledWavelengths wl = wavelengths();
    BsdfSample sample{};
    for (int attempt = 0; attempt < 64; ++attempt) {
        const float u = (attempt + 0.5f) / 64.0f;
        sample = nr::openpbr::sample(
            dispersiveMaterial, normal, normal, view, glm::vec3(0.5f, 0.5f, u),
            wl, unitThroughput);
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
