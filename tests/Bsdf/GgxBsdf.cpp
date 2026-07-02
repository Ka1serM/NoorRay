#include "BsdfFixture.h"

#include <catch2/catch_test_macros.hpp>

class GgxBsdfTest : public BsdfTestFixture {};

TEST_CASE_METHOD(GgxBsdfTest,
    "GGX reflection PDF numerically integrates to a probability", "[bsdf][ggx]")
{
    const BsdfMaterialParameters material = makeMaterial(
        1.0f, 1.0f, 0.5f, 0.35f);
    constexpr int thetaSamples = 256;
    constexpr int phiSamples = 256;
    double integral = 0.0;
    for (int y = 0; y < thetaSamples; ++y) {
        for (int x = 0; x < phiSamples; ++x) {
            const float z = (y + 0.5f) / thetaSamples;
            const float phi = 2.0f * nr::bsdf::Pi * (x + 0.5f) / phiSamples;
            const float radius = std::sqrt(1.0f - z * z);
            const glm::vec3 light(radius * std::cos(phi), radius * std::sin(phi), z);
            integral += nr::bsdf::pdf(material, normal, normal, view, light);
        }
    }
    integral *= 2.0 * nr::bsdf::Pi / static_cast<double>(thetaSamples * phiSamples);
    CHECK(integral > 0.97);
    CHECK(integral <= 1.01);
}
