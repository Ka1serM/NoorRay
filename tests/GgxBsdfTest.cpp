#include "BsdfTestFixture.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

class GgxBsdfTest : public BsdfTestFixture {};

TEST_CASE_METHOD(GgxBsdfTest,
    "GGX reflection PDF numerically integrates to a probability", "[bsdf][ggx]")
{
    constexpr int thetaSamples = 256;
    constexpr int phiSamples = 256;
    double integral = 0.0;
    for (int y = 0; y < thetaSamples; ++y) {
        for (int x = 0; x < phiSamples; ++x) {
            const glm::vec3 light = nr::sampling::uniformHemisphere({
                (y + 0.5f) / thetaSamples, (x + 0.5f) / phiSamples});
            const glm::vec3 halfVector = glm::normalize(view + light);
            integral += nr::shading::ggx::reflectionPdf(
                normal, view, halfVector, 0.35f);
        }
    }
    integral *= 2.0 * nr::shading::ggx::Pi
        / static_cast<double>(thetaSamples * phiSamples);
    CHECK(integral > 0.97);
    CHECK(integral <= 1.01);
}

TEST_CASE_METHOD(GgxBsdfTest,
    "GGX remains finite immediately above the singular cutoff", "[bsdf][ggx]")
{
    constexpr float roughness = 0.004f;
    const float pdf = nr::shading::ggx::reflectionPdf(
        normal, view, normal, roughness);
    const float expected = 1.0f / (4.0f * nr::shading::ggx::Pi
        * roughness * roughness * roughness * roughness);

    CHECK(std::isfinite(pdf));
    CHECK(pdf == Catch::Approx(expected).epsilon(2e-4f));
    CHECK(pdf > 1e8f);
}
