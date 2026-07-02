#include "BsdfFixture.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

class BsdfConventionTest : public BsdfTestFixture {};

TEST_CASE_METHOD(BsdfConventionTest,
    "evaluation includes the incident cosine", "[bsdf][convention]")
{
    const BsdfMaterialParameters lambert = makeMaterial(
        1.0f, 0.0f, 0.0f, 0.0f);
    const SampledWavelengths wl = wavelengths();
    RandomState rng = seedRandom(42);

    const float normalValue = nr::bsdf::evaluate(
        lambert, normal, normal, view, normal, wl, rng)[0];
    constexpr float cosine = 0.25f;
    const glm::vec3 grazing(std::sqrt(1.0f - cosine * cosine), 0.0f, cosine);
    const float grazingValue = nr::bsdf::evaluate(
        lambert, normal, normal, view, grazing, wl, rng)[0];

    CHECK(normalValue == Catch::Approx(1.0f / nr::bsdf::Pi).margin(2e-5f));
    CHECK(grazingValue == Catch::Approx(cosine / nr::bsdf::Pi).margin(2e-5f));
    CHECK(nr::bsdf::evaluate(
        lambert, normal, normal, view, -normal, wl, rng)[0] == 0.0f);
}

TEST_CASE_METHOD(BsdfConventionTest,
    "sample weight equals cosine-weighted evaluation over PDF",
    "[bsdf][convention]")
{
    const BsdfMaterialParameters material = makeMaterial(
        0.65f, 0.2f, 0.5f, 0.35f);
    const SampledWavelengths wl = wavelengths();
    int validSamples = 0;

    for (uint64_t i = 0; i < 1024; ++i)
    {
        RandomState rng = seedRandom(i);
        const BsdfSample sample = nr::bsdf::sample(
            material, normal, normal, view, rng, wl);
        if (sample.pdf == 0.0f)
            continue;
        ++validSamples;

        const SampledSpectrum value = nr::bsdf::evaluate(
            material, normal, normal, view, sample.direction, wl, rng);
        for (int channel = 0; channel < NrSpectrumSamples; ++channel)
            CHECK(sample.weight[channel] ==
                Catch::Approx(value[channel] / sample.pdf).epsilon(3e-5f));
    }

    CHECK(validSamples > 1000);
}

TEST_CASE_METHOD(BsdfConventionTest,
    "face-forwards an outward direction convention on back faces",
    "[bsdf][convention]")
{
    const BsdfMaterialParameters material = makeMaterial(
        0.8f, 0.0f, 0.5f, 0.4f);
    const SampledWavelengths wl = wavelengths();
    const glm::vec3 backView = -view;

    for (uint64_t i = 0; i < 128; ++i)
    {
        RandomState rng = seedRandom(i);
        const BsdfSample sample = nr::bsdf::sample(
            material, normal, normal, backView, rng, wl);
        REQUIRE(sample.pdf > 0.0f);
        CHECK(sample.direction.z < 0.0f);
    }
}
