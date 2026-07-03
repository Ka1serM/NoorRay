#include "BsdfTestFixture.h"

#include <catch2/catch_test_macros.hpp>

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
            integral += Bsdf::pdfGgxReflection(view, normal, halfVector, 0.35f);
        }
    }
    integral *= 2.0 * BsdfPi / static_cast<double>(thetaSamples * phiSamples);
    CHECK(integral > 0.97);
    CHECK(integral <= 1.01);
}
