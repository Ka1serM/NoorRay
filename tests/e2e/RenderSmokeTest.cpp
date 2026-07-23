#include "RenderTestFixture.h"

#include "IO/BitmapReader.h"

#include <catch2/catch_test_macros.hpp>

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
