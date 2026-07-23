#include "BsdfTestFixture.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

class OpaqueBsdfTest : public BsdfTestFixture {};

TEST_CASE_METHOD(OpaqueBsdfTest,
    "Lambert and dielectric GGX integrate without creating energy", "[bsdf][furnace]")
{
    const float blackDiffuse = integrateOpaque(0.0f, 0.0f, 0.0f, 0.5f);
    const float whiteDiffuse = integrateOpaque(1.0f, 0.0f, 0.0f, 0.5f);
    const float whiteDielectric = integrateOpaque(1.0f, 0.0f, 0.5f, 0.5f);

    CHECK(blackDiffuse == Catch::Approx(0.0f).margin(5e-5f));
    CHECK(whiteDiffuse == Catch::Approx(1.0f).margin(0.003f));
    CHECK(whiteDielectric > 0.90f);
    CHECK(whiteDielectric <= 1.01f);
}

TEST_CASE_METHOD(OpaqueBsdfTest,
    "metallic GGX hemispherical reflectance is bounded", "[bsdf][ggx]")
{
    const float reflectance = integrateOpaque(0.8f, 1.0f, 0.5f, 0.35f);
    CHECK(reflectance > 0.65f);
    CHECK(reflectance <= 0.81f);
}

TEST_CASE_METHOD(OpaqueBsdfTest,
    "perfectly smooth metal samples only delta reflection", "[bsdf][metal][delta]")
{
    constexpr float reflectance = 0.8f;
    const Bsdf bsdf(
        normal, normal, view, SampledSpectrum(reflectance),
        1.0f, 0.5f, 0.0f, 0.0f, SampledSpectrum(1.0f),
        SellmeierCoefficients{}, SampledWavelengths::sampleUniform(0.5f));

    RandomState rng = seedRandom(0x51ee7u);
    for (int i = 0; i < 128; ++i) {
        const BsdfSample sample = bsdf.sample(rng);
        CHECK(sample.event == BsdfEvent::Specular);
        CHECK(sample.direction.x == Catch::Approx(0.0f).margin(1e-6f));
        CHECK(sample.direction.y == Catch::Approx(0.0f).margin(1e-6f));
        CHECK(sample.direction.z == Catch::Approx(1.0f).margin(1e-6f));
        CHECK(sample.weight[0] == Catch::Approx(reflectance).margin(1e-5f));
        CHECK(sample.pdf == Catch::Approx(nr::shading::ggx::SingularPdf));
    }
    CHECK(bsdf.evaluate(glm::vec3(0.0f, 0.0f, 1.0f)).pdf == 0.0f);
}

TEST_CASE_METHOD(OpaqueBsdfTest,
    "perfectly smooth dielectric keeps its continuous diffuse closure",
    "[bsdf][diffuse][delta][regression]")
{
    const Bsdf bsdf(
        normal, normal, view, SampledSpectrum(0.8f),
        0.0f, 0.5f, 0.0f, 0.0f, SampledSpectrum(1.0f),
        SellmeierCoefficients{}, SampledWavelengths::sampleUniform(0.5f));

    RandomState rng = seedRandom(0xd1ffu);
    double average = 0.0;
    int diffuseSamples = 0;
    int specularSamples = 0;
    bool allSamplesContribute = true;
    for (int i = 0; i < 16384; ++i)
    {
        const BsdfSample sample = bsdf.sample(rng);
        average += sample.weight[0];
        allSamplesContribute = allSamplesContribute
            && sample.pdf > 0.0f && sample.weight[0] > 0.0f
            && std::isfinite(sample.weight[0]);
        diffuseSamples += sample.event == BsdfEvent::Diffuse;
        specularSamples += sample.event == BsdfEvent::Specular;
    }

    average /= 16384.0;
    INFO("average=" << average << " diffuse=" << diffuseSamples
        << " specular=" << specularSamples);
    CHECK(allSamplesContribute);
    CHECK(diffuseSamples > 14000);
    CHECK(specularSamples > 500);
    CHECK(average > 0.7);
}

TEST_CASE_METHOD(OpaqueBsdfTest,
    "GGX uses Cycles' almost-specular threshold without a roughness floor", "[bsdf][ggx][delta]")
{
    CHECK(nr::shading::ggx::isAlmostSpecular(0.003f));
    CHECK_FALSE(nr::shading::ggx::isAlmostSpecular(0.004f));

    const Bsdf bsdf(
        normal, normal, view, SampledSpectrum(0.8f),
        1.0f, 0.5f, 0.01f, 0.0f, SampledSpectrum(1.0f),
        SellmeierCoefficients{}, SampledWavelengths::sampleUniform(0.5f));
    CHECK(bsdf.roughness() == Catch::Approx(0.01f));
}

TEST_CASE_METHOD(OpaqueBsdfTest,
    "low-roughness metal never samples a zero-energy diffuse lobe", "[bsdf][metal][ggx]")
{
    const Bsdf bsdf(
        normal, normal, view, SampledSpectrum(0.8f),
        1.0f, 0.5f, 0.01f, 0.0f, SampledSpectrum(1.0f),
        SellmeierCoefficients{}, SampledWavelengths::sampleUniform(0.5f));

    RandomState rng = seedRandom(0x10b3u);
    for (int i = 0; i < 1024; ++i) {
        const BsdfSample sample = bsdf.sample(rng);
        CHECK(sample.event == BsdfEvent::Specular);
        CHECK(sample.pdf > 0.0f);
        CHECK(std::isfinite(sample.pdf));
        CHECK(std::isfinite(sample.weight[0]));
        CHECK(sample.weight[0] > 0.0f);
    }
}

TEST_CASE_METHOD(OpaqueBsdfTest,
    "metal throughput stays bounded across the narrow GGX transition",
    "[bsdf][metal][ggx][regression]")
{
    for (const float roughness : {0.003f, 0.004f, 0.009f, 0.010f, 0.011f, 0.02f})
    {
        const Bsdf bsdf(
            normal, normal, view, SampledSpectrum(0.8f),
            1.0f, 0.5f, roughness, 0.0f, SampledSpectrum(1.0f),
            SellmeierCoefficients{}, SampledWavelengths::sampleUniform(0.5f));
        RandomState rng = seedRandom(0x10u);
        double average = 0.0;
        float maximum = 0.0f;
        float maximumPdf = 0.0f;
        float maximumEval = 0.0f;
        glm::vec3 maximumDirection{};
        int accepted = 0;
        bool allFinite = true;
        for (int i = 0; i < 65536; ++i)
        {
            const BsdfSample sample = bsdf.sample(rng);
            allFinite = allFinite && std::isfinite(sample.weight[0]);
            if (sample.pdf <= 0.0f)
                continue;
            average += sample.weight[0];
            if (sample.weight[0] > maximum)
            {
                maximum = sample.weight[0];
                const BsdfEvaluation evaluation = bsdf.evaluate(
                    sample.direction);
                maximumPdf = evaluation.pdf;
                maximumEval = evaluation.value[0];
                maximumDirection = sample.direction;
            }
            ++accepted;
        }
        INFO("roughness=" << roughness << " average="
            << average / fmaxf(static_cast<float>(accepted), 1.0f)
            << " maximum=" << maximum << " pdf=" << maximumPdf
            << " eval=" << maximumEval << " direction=("
            << maximumDirection.x << ',' << maximumDirection.y << ','
            << maximumDirection.z << ") accepted=" << accepted);
        CHECK(allFinite);
        CHECK(accepted > 65000);
        CHECK(maximum <= 0.801f);
    }
}

TEST_CASE_METHOD(OpaqueBsdfTest, "white furnace remains energy conserving at grazing angles", "[bsdf][furnace]")
{
    for (const float cosine : {0.5f, 0.173648f, 0.087156f}) {
        const glm::vec3 grazingView(std::sqrt(1.0f - cosine * cosine), 0.0f, cosine);
        const float reflectance = integrateOpaque(1.0f, 0.0f, 0.5f, 0.5f, grazingView);
        INFO("view cosine=" << cosine << " reflectance=" << reflectance);
        CHECK(reflectance >= 0.995f);
        CHECK(reflectance <= 1.01f);
    }
}

TEST_CASE_METHOD(OpaqueBsdfTest, "shading normals keep reflections above the geometric surface", "[bsdf][normal]")
{
    const glm::vec3 geometricNormal(0.0f, 0.0f, 1.0f);
    const glm::vec3 view = glm::normalize(glm::vec3(0.999f, 0.0f, 0.045f));
    const glm::vec3 divergentNormal = glm::normalize(glm::vec3(-0.12f, 0.0f, 0.993f));
    REQUIRE(glm::dot(geometricNormal, glm::reflect(-view, divergentNormal)) < 0.0f);

    const glm::vec3 corrected = Bsdf::clampShadingNormal(
        geometricNormal, divergentNormal, view);
    const glm::vec3 reflected = glm::reflect(-view, corrected);

    CHECK(glm::dot(geometricNormal, reflected) > 0.0f);
    CHECK(glm::dot(corrected, view) > 0.0f);
    CHECK(glm::length(corrected) == Catch::Approx(1.0f).margin(1e-5f));

    const glm::vec3 validNormal = glm::normalize(glm::vec3(0.12f, 0.0f, 0.993f));
    const glm::vec3 unchanged = Bsdf::clampShadingNormal(
        geometricNormal, validNormal, view);
    CHECK(glm::dot(unchanged, validNormal) == Catch::Approx(1.0f).margin(1e-5f));
}
