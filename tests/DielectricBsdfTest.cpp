#include "BsdfTestFixture.h"

#include "Shading/Material.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

class DielectricBsdfTest : public BsdfTestFixture {};

TEST_CASE("Fresnel formulas are reusable independently of Bsdf", "[bsdf][fresnel]")
{
    constexpr float f0 = 0.04f;
    const float ior = nr::shading::dielectric::iorFromNormalReflectance(f0);

    CHECK(ior == Catch::Approx(1.5f));
    CHECK(nr::shading::dielectric::fresnel(1.0f, 1.0f, ior)
        == Catch::Approx(f0));
    CHECK(nr::shading::dielectric::fresnelFromNormalReflectance(1.0f, f0)
        == Catch::Approx(f0));
}

TEST_CASE("dispersive glass requires a hero wavelength before BSDF evaluation",
    "[bsdf][dielectric][dispersion][regression]")
{
    SampledWavelengths wavelengths = SampledWavelengths::sampleUniform(0.17f);

    Material constantGlass;
    constantGlass.transmission = 1.0f;
    constantGlass.sellmeier = constantIorSellmeier(1.5f);
    CHECK_FALSE(constantGlass.hasDispersiveIor(wavelengths));

    Material prism;
    prism.transmission = 1.0f;
    prism.sellmeier = fitSellmeierFromFraunhofer({1.50f, 1.52f, 1.54f});
    REQUIRE(prism.hasDispersiveIor(wavelengths));

    wavelengths.terminateSecondary();
    CHECK(wavelengths.secondaryTerminated());
    CHECK_FALSE(prism.hasDispersiveIor(wavelengths));
}

TEST_CASE("constant-IOR transmission retains the spectral packet",
    "[bsdf][dielectric][dispersion][regression]")
{
    PathState state{};
    state.wl = SampledWavelengths::sampleUniform(0.31f);
    state.etaScale = 1.0f;

    state.transmit(1.5f);

    CHECK_FALSE(state.wl.secondaryTerminated());
    CHECK(state.etaScale == Catch::Approx(2.25f));
}

TEST_CASE_METHOD(DielectricBsdfTest,
    "microfacet dielectric sampling produces valid reflection and transmission",
    "[bsdf][dielectric]")
{
    constexpr int sampleCount = 100000;
    const Bsdf bsdf(
        normal, normal, view, SampledSpectrum(0.0f),
        0.0f, 0.5f, 0.045f, 1.0f, SampledSpectrum(1.0f),
        constantIorSellmeier(1.5f),
        SampledWavelengths::sampleUniform(0.5f));
    RandomState rng = seedRandom(0x5eedu);
    int reflections = 0;
    int rejected = 0;
    for (int i = 0; i < sampleCount; ++i) {
        const BsdfSample sample = bsdf.sample(rng);
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

TEST_CASE_METHOD(DielectricBsdfTest,
    "closure sampling returns the combined spectral BSDF and PDF",
    "[bsdf][dielectric][closure]")
{
    const Bsdf bsdf(
        normal, normal, view, SampledSpectrum(0.7f),
        0.1f, 0.5f, 0.25f, 0.65f, SampledSpectrum(0.9f),
        constantIorSellmeier(1.5f), SampledWavelengths::sampleUniform(0.37f));

    RandomState rng = seedRandom(0xc105u);
    int reflected = 0;
    int transmitted = 0;
    for (int i = 0; i < 4096; ++i)
    {
        const BsdfSample sample = bsdf.sample(rng);
        if (sample.pdf <= 0.0f)
            continue;

        const BsdfEvaluation evaluation = bsdf.evaluate(sample.direction);
        const float combinedPdf = evaluation.pdf;
        REQUIRE(combinedPdf > 0.0f);
        CHECK(sample.pdf == Catch::Approx(combinedPdf).epsilon(2.0e-4f));
        const SampledSpectrum expected = evaluation.value
            * (fabsf(glm::dot(normal, sample.direction)) / combinedPdf);
        for (int wavelength = 0; wavelength < NrSpectrumSamples; ++wavelength)
            CHECK(sample.weight[wavelength]
                == Catch::Approx(expected[wavelength]).epsilon(3.0e-4f));

        if (sample.event == BsdfEvent::Transmission)
            ++transmitted;
        else
            ++reflected;
    }
    CHECK(reflected > 0);
    CHECK(transmitted > 0);
}

TEST_CASE_METHOD(DielectricBsdfTest,
    "smooth dielectric uses singular Fresnel-weighted PDFs", "[bsdf][dielectric][delta]")
{
    RandomState rng = seedRandom(0xd31au);
    constexpr float fresnel = 0.04f;
    const Bsdf bsdf(
        normal, normal, view, SampledSpectrum(0.0f),
        0.0f, 0.5f, 0.0f, 1.0f, SampledSpectrum(1.0f),
        constantIorSellmeier(1.5f),
        SampledWavelengths::sampleUniform(0.5f));
    for (int i = 0; i < 256; ++i) {
        const BsdfSample sample = bsdf.sample(rng);
        if (sample.event == BsdfEvent::Specular) {
            CHECK(sample.direction.z == Catch::Approx(1.0f));
            CHECK(sample.pdf == Catch::Approx(
                fresnel * nr::shading::ggx::SingularPdf).margin(1.0f));
            CHECK(sample.weight[0] == Catch::Approx(1.0f));
        } else {
            REQUIRE(sample.event == BsdfEvent::Transmission);
            CHECK(sample.direction.z == Catch::Approx(-1.0f));
            CHECK(sample.pdf == Catch::Approx(
                (1.0f - fresnel) * nr::shading::ggx::SingularPdf).margin(1.0f));
            CHECK(sample.weight[0] == Catch::Approx(1.0f / (1.5f * 1.5f)));
        }
    }
}

TEST_CASE_METHOD(DielectricBsdfTest,
    "smooth dielectric follows Snell's law when entering and exiting",
    "[bsdf][dielectric][ior][regression]")
{
    constexpr float ior = 1.5f;
    constexpr float incidentSine = 0.5f;
    constexpr float incidentCosine = 0.8660254038f;
    const auto makeGlass = [&](const glm::vec3 testView) {
        return Bsdf(normal, normal, testView, SampledSpectrum(0.0f),
            0.0f, 0.5f, 0.0f, 1.0f, SampledSpectrum(1.0f),
            constantIorSellmeier(ior),
            SampledWavelengths::sampleUniform(0.5f));
    };
    const auto findTransmission = [](const Bsdf& bsdf) {
        RandomState rng = seedRandom(0x10fu);
        for (int attempt = 0; attempt < 256; ++attempt)
        {
            const BsdfSample sample = bsdf.sample(rng);
            if (sample.event == BsdfEvent::Transmission)
                return sample;
        }
        return BsdfSample{};
    };

    SECTION("air to glass")
    {
        const glm::vec3 incident(incidentSine, 0.0f, -incidentCosine);
        const BsdfSample sample = findTransmission(makeGlass(-incident));
        REQUIRE(sample.event == BsdfEvent::Transmission);
        const float transmittedSine = incidentSine / ior;
        CHECK(sample.direction.x == Catch::Approx(transmittedSine));
        CHECK(sample.direction.z == Catch::Approx(
            -sqrtf(1.0f - transmittedSine * transmittedSine)));
        CHECK(sample.eta == Catch::Approx(ior));
    }

    SECTION("glass to air")
    {
        const glm::vec3 incident(incidentSine, 0.0f, incidentCosine);
        const BsdfSample sample = findTransmission(makeGlass(-incident));
        REQUIRE(sample.event == BsdfEvent::Transmission);
        const float transmittedSine = incidentSine * ior;
        CHECK(sample.direction.x == Catch::Approx(transmittedSine));
        CHECK(sample.direction.z == Catch::Approx(
            sqrtf(1.0f - transmittedSine * transmittedSine)));
        CHECK(sample.eta == Catch::Approx(1.0f / ior));
    }
}

TEST_CASE_METHOD(DielectricBsdfTest,
    "rough glass restores masked reflection and transmission energy",
    "[bsdf][dielectric][furnace][regression]")
{
    constexpr int sampleCount = 256;
    constexpr float ior = 1.5f;
    const SampledWavelengths testWavelengths =
        SampledWavelengths::sampleUniform(0.5f);

    for (const bool exiting : {false, true})
    {
        const glm::vec3 testView = exiting ? -normal : normal;
        const Bsdf bsdf(
            normal, normal, testView, SampledSpectrum(0.0f),
            0.0f, 0.5f, 1.0f, 1.0f, SampledSpectrum(1.0f),
            constantIorSellmeier(ior), testWavelengths);
        const float etaPath = exiting ? 1.0f / ior : ior;
        double energy = 0.0;
        for (int hemisphere = 0; hemisphere < 2; ++hemisphere)
        {
            for (int y = 0; y < sampleCount; ++y)
            {
                for (int x = 0; x < sampleCount; ++x)
                {
                    glm::vec3 light = nr::sampling::uniformHemisphere({
                        (y + 0.5f) / sampleCount,
                        (x + 0.5f) / sampleCount});
                    if (hemisphere != 0)
                        light = -light;
                    const bool transmitted = glm::dot(testView, light) < 0.0f;
                    const float radianceToEnergy = transmitted
                        ? etaPath * etaPath : 1.0f;
                    energy += bsdf.evaluate(light).value[0] * fabsf(light.z)
                        * radianceToEnergy;
                }
            }
        }
        energy *= 2.0 * nr::shading::ggx::Pi
            / static_cast<double>(sampleCount * sampleCount);
        INFO("exiting=" << exiting << " energy=" << energy);
        CHECK(energy >= 0.98);
        CHECK(energy <= 1.02);
    }
}
