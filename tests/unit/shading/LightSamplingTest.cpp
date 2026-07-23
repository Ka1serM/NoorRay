#include "Light/DirectionalLight.h"
#include "Light/RectLight.h"
#include "Light/SpotLight.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

TEST_CASE("spot light soft radius samples a spherical emitter", "[light]")
{
    SpotLight light{};
    light.position = glm::vec3(1.0f, 2.0f, 3.0f);
    light.direction = glm::vec3(0.0f, 0.0f, -1.0f);
    light.color = glm::vec3(0.0f);
    light.softRadius = 2.0f;

    const glm::vec3 surfacePosition(1.0f, 2.0f, -5.0f);
    RandomState rng = seedRandom(42u);
    const LightSample sample = light.sampleLi(
        surfacePosition, rng, SampledWavelengths::sampleUniform(0.5f),
        nullptr, nullptr, nullptr);

    const glm::vec3 sampledPosition =
        surfacePosition + sample.direction * sample.distance;
    CHECK(glm::length(sampledPosition - light.position)
        == Catch::Approx(light.softRadius).margin(1e-5f));
}

TEST_CASE("spherical rectangle sampling has a constant solid-angle PDF", "[light][rect]")
{
    RectLight light{};
    light.position = glm::vec3(0.0f, 0.0f, 1.0f);
    light.direction = glm::vec3(0.0f, 0.0f, -1.0f);
    light.tangent = glm::vec3(1.0f, 0.0f, 0.0f);
    light.color = glm::vec3(0.0f);
    light.width = 2.0f;
    light.height = 2.0f;

    constexpr float expectedSolidAngle = 2.09439510239319549f;
    RandomState rng = seedRandom(0xa4eau);
    for (int i = 0; i < 64; ++i)
    {
        const LightSample sample = light.sampleLi(
            glm::vec3(0.0f), rng, SampledWavelengths::sampleUniform(0.5f),
            nullptr, nullptr, nullptr);
        const glm::vec3 point = sample.direction * sample.distance;
        CHECK(fabsf(point.x) <= 1.00001f);
        CHECK(fabsf(point.y) <= 1.00001f);
        CHECK(point.z == Catch::Approx(1.0f).margin(2.0e-5f));
        CHECK(sample.pdf == Catch::Approx(1.0f / expectedSolidAngle)
            .epsilon(2.0e-5f));
    }
}

TEST_CASE("directional light treats soft angle as an angular diameter", "[light][sun]")
{
    DirectionalLight light{};
    light.direction = glm::vec3(0.0f, 0.0f, -1.0f);
    light.color = glm::vec3(0.0f);
    light.softAngle = 0.53f;

    RandomState rng = seedRandom(0x5a1u);
    const LightSample sample = light.sampleLi(
        glm::vec3(0.0f), rng, SampledWavelengths::sampleUniform(0.5f),
        nullptr, nullptr, nullptr);
    const float halfAngle = 0.5f * light.softAngle * LightPi / 180.0f;
    const float expectedPdf = 1.0f
        / (2.0f * LightPi * oneMinusCosine(halfAngle));

    CHECK(glm::dot(sample.direction, -light.direction) >= cosf(halfAngle));
    CHECK(sample.pdf == Catch::Approx(expectedPdf).epsilon(2.0e-5f));
}

TEST_CASE("tiny sphere lights retain a finite solid-angle PDF", "[light][point]")
{
    constexpr float radius = 1.0e-4f;
    constexpr float distance = 10.0f;
    const float pdf = sphereLightPdf(
        glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, distance), radius);
    const float smallAngleApproximation = distance * distance
        / (LightPi * radius * radius);

    CHECK(std::isfinite(pdf));
    CHECK(pdf == Catch::Approx(smallAngleApproximation).epsilon(2.0e-5f));
}

TEST_CASE("zero-radius spot light remains a point emitter", "[light]")
{
    SpotLight light{};
    light.position = glm::vec3(0.0f, 0.0f, 5.0f);
    light.direction = glm::vec3(0.0f, 0.0f, -1.0f);
    light.color = glm::vec3(0.0f);

    RandomState rng = seedRandom(42u);
    const RandomState initialRng = rng;
    const LightSample sample = light.sampleLi(
        glm::vec3(0.0f), rng, SampledWavelengths::sampleUniform(0.5f),
        nullptr, nullptr, nullptr);

    CHECK(sample.distance == Catch::Approx(5.0f));
    CHECK(sample.direction.x == Catch::Approx(0.0f));
    CHECK(sample.direction.y == Catch::Approx(0.0f));
    CHECK(sample.direction.z == Catch::Approx(1.0f));
    CHECK(rng == initialRng);
}
