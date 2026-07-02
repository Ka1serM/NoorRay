#include "RenderTestFixture.h"

#include "IO/BitmapReader.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

class RenderSmokeTest : public RenderTestFixture {};

TEST_CASE_METHOD(RenderSmokeTest, "NoorRay renders a simple sphere to EXR", "[e2e]")
{
    const std::string output = render("simple_sphere.json", 4, "simple_sphere.exr");
    const Bitmap bitmap = BitmapReader::read(output);
    CHECK(bitmap.width() == 64);
    CHECK(bitmap.height() == 64);
    CHECK(bitmap.pixels().size() == 64 * 64);

    bool finite = true;
    double rgbSum = 0.0;
    float maxRgb = 0.0f;
    for (const glm::vec4& pixel : bitmap.pixels())
    {
        for (int channel = 0; channel < 3; ++channel)
        {
            finite = finite && std::isfinite(pixel[channel]);
            rgbSum += pixel[channel];
            if (pixel[channel] > maxRgb)
                maxRgb = pixel[channel];
        }
    }
    const double meanRgb = rgbSum / (3.0 * bitmap.pixels().size());
    INFO("mean RGB=" << meanRgb << ", max RGB=" << maxRgb);
    CHECK(finite);
    CHECK(meanRgb > 1e-4);
    CHECK(maxRgb > 1e-4f);
}
