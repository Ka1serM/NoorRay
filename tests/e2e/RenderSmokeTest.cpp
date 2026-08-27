#include "RenderTestFixture.h"

#include "IO/BitmapReader.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

class RenderSmokeTest : public RenderTestFixture {};

TEST_CASE_METHOD(RenderSmokeTest, "NoorRay renders a simple sphere to EXR", "[e2e]")
{
    const std::string output = render("simple_sphere.nrscene", 2, "simple_sphere.exr");
    const Bitmap bitmap = BitmapReader::read(output);
    CHECK(bitmap.width() == 256);
    CHECK(bitmap.height() == 256);
    CHECK(bitmap.pixels().size() == 256 * 256);
}

TEST_CASE_METHOD(RenderSmokeTest, "NoorRay renders a PBRT scene to EXR", "[e2e][pbrt]")
{
    const std::string output = render("simple_sphere.pbrt", 1, "simple_sphere_pbrt.exr");
    const Bitmap bitmap = BitmapReader::read(output);
    CHECK(bitmap.width() == 256);
    CHECK(bitmap.height() == 256);
    CHECK(bitmap.pixels().size() == 256 * 256);
}

TEST_CASE_METHOD(RenderSmokeTest,
    "Vulkan path tracer shades a surface with an analytic point light",
    "[e2e][light]")
{
    const std::string output = render("point_light_sphere.pbrt", 16,
        "point_light_sphere.exr", 128, 128, 41);
    const Bitmap bitmap = BitmapReader::read(output);
    REQUIRE(bitmap.width() == 128);
    REQUIRE(bitmap.height() == 128);

    float peak = 0.0f;
    double sum = 0.0;
    for (const glm::vec4 pixel : bitmap.pixels())
    {
        const float luminance = 0.2126f * pixel.r + 0.7152f * pixel.g
            + 0.0722f * pixel.b;
        REQUIRE(std::isfinite(luminance));
        peak = std::max(peak, luminance);
        sum += luminance;
    }
    const float mean = static_cast<float>(sum / bitmap.pixels().size());
    INFO("analytic-light mean=" << mean << ", peak=" << peak);
    CHECK(mean > 0.01f);
    CHECK(peak > 0.1f);
}

TEST_CASE_METHOD(RenderSmokeTest,
    "Vulkan path tracer samples emissive mesh triangles through SVM",
    "[e2e][light][mesh-light]")
{
    const std::string output = render("mesh_light_sphere.pbrt", 16,
        "mesh_light_sphere.exr", 96, 96, 73);
    const Bitmap bitmap = BitmapReader::read(output);
    REQUIRE(bitmap.width() == 96);
    REQUIRE(bitmap.height() == 96);
    float peak = 0.0f;
    double sum = 0.0;
    for (const glm::vec4 pixel : bitmap.pixels()) {
        const float luminance = 0.2126f * pixel.r + 0.7152f * pixel.g
            + 0.0722f * pixel.b;
        REQUIRE(std::isfinite(luminance));
        peak = std::max(peak, luminance);
        sum += luminance;
    }
    const float mean = static_cast<float>(sum / bitmap.pixels().size());
    INFO("mesh-light mean=" << mean << ", peak=" << peak);
    CHECK(mean > 0.005f);
    CHECK(peak > 0.05f);
}
