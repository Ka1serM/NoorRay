#include "RenderTestFixture.h"

#include "IO/BitmapReader.h"

#include <catch2/catch_test_macros.hpp>

class RenderSmokeTest : public RenderTestFixture {};

TEST_CASE_METHOD(RenderSmokeTest, "NoorRay renders a simple sphere to EXR", "[e2e]")
{
    const std::string output = render("simple_sphere.json", 4, "simple_sphere.exr");
    const Bitmap bitmap = BitmapReader::read(output);
    CHECK(bitmap.width() == 64);
    CHECK(bitmap.height() == 64);
    CHECK(bitmap.pixels().size() == 64 * 64);
}
