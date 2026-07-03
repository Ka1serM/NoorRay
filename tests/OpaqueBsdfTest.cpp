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
