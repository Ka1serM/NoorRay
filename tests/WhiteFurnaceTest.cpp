#include "RenderTestFixture.h"

#include "IO/BitmapReader.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

class WhiteFurnaceTest : public RenderTestFixture {};

TEST_CASE_METHOD(WhiteFurnaceTest, "white furnace conserves energy", "[e2e][furnace]")
{
    const std::string output = render("white_furnace.json", 1024, "white_furnace.exr");
    const Bitmap bitmap = BitmapReader::read(output);
    REQUIRE(bitmap.width() == 64);
    REQUIRE(bitmap.height() == 64);

    std::array<float, 3> background{};
    for (int channel = 0; channel < 3; ++channel) {
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x)
                background[channel] += bitmap.pixel(x, y)[channel] / 64.0f;
        REQUIRE(background[channel] > 0.0f);

        // A perfectly white object under a uniform white environment must be
        // indistinguishable from the background across the entire image.
        double sum = 0.0;
        std::vector<float> values;
        for (uint32_t y = 0; y < bitmap.height(); ++y) {
            for (uint32_t x = 0; x < bitmap.width(); ++x) {
                const float value = bitmap.pixel(x, y)[channel];
                sum += value;
                values.push_back(value);
            }
        }

        std::ranges::sort(values);
        const size_t trim = values.size() / 100;
        double trimmedSquaredError = 0.0;
        for (size_t index = trim; index < values.size() - trim; ++index) {
            const double error = values[index] - background[channel];
            trimmedSquaredError += error * error;
        }
        const float mean = static_cast<float>(sum / values.size());
        const float firstPercentile = values[trim];
        const float ninetyNinthPercentile = values[values.size() - trim - 1];
        const float trimmedRelativeRms = static_cast<float>(std::sqrt(trimmedSquaredError / (values.size() - 2 * trim)) / background[channel]);
        INFO("channel=" << channel << " mean=" << mean << " background=" << background[channel] << " p01=" << firstPercentile
             << " p99=" << ninetyNinthPercentile << " trimmed relative RMS=" << trimmedRelativeRms);
        CHECK(mean >= background[channel] * 0.97f);
        CHECK(mean <= background[channel] * 1.03f);
        CHECK(firstPercentile >= background[channel] * 0.95f);
        CHECK(ninetyNinthPercentile <= background[channel] * 1.05f);
        CHECK(trimmedRelativeRms <= 0.02f);
    }
}
