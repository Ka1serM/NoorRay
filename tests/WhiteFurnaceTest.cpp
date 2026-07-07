#include "RenderTestFixture.h"

#include "IO/BitmapReader.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>

class WhiteFurnaceTest : public RenderTestFixture {};

namespace {
void checkEnergyConservation(const Bitmap& bitmap)
{
    REQUIRE(bitmap.width() == 1920);
    REQUIRE(bitmap.height() == 1080);

    std::array<float, 3> background{};
    constexpr uint32_t patchSize = 128;
    for (int channel = 0; channel < 3; ++channel) {
        double backgroundSum = 0.0;
        for (uint32_t y = 0; y < patchSize; ++y) {
            for (uint32_t x = 0; x < patchSize; ++x) {
                backgroundSum += bitmap.pixel(x, y)[channel];
                backgroundSum += bitmap.pixel(bitmap.width() - 1 - x, y)[channel];
                backgroundSum += bitmap.pixel(x, bitmap.height() - 1 - y)[channel];
                backgroundSum += bitmap.pixel(
                    bitmap.width() - 1 - x, bitmap.height() - 1 - y)[channel];
            }
        }
        background[channel] = static_cast<float>(
            backgroundSum / (4.0 * patchSize * patchSize));
        REQUIRE(background[channel] > 0.0f);

        double sum = 0.0;
        double squaredError = 0.0;
        for (uint32_t y = 0; y < bitmap.height(); ++y) {
            for (uint32_t x = 0; x < bitmap.width(); ++x) {
                const float value = bitmap.pixel(x, y)[channel];
                sum += value;
                const double relativeError = value / background[channel] - 1.0;
                squaredError += relativeError * relativeError;
            }
        }

        const double pixelCount = static_cast<double>(bitmap.width()) * bitmap.height();
        const float relativeBias = static_cast<float>(sum / pixelCount / background[channel] - 1.0);
        const float relativeRmse = static_cast<float>(std::sqrt(squaredError / pixelCount));
        INFO("channel=" << channel << " background=" << background[channel]
             << " relative bias=" << relativeBias << " relative RMSE=" << relativeRmse);
        CHECK(std::abs(relativeBias) <= 0.001f);
        CHECK(relativeRmse <= 0.0045f);
    }
}
}

TEST_CASE_METHOD(WhiteFurnaceTest, "white furnace conserves energy", "[e2e][furnace]")
{
    const std::string output = render("white_furnace.nrscene", 4096, "white_furnace.exr");
    checkEnergyConservation(BitmapReader::read(output));
}

TEST_CASE_METHOD(WhiteFurnaceTest, "white furnace teapot conserves energy at silhouettes", "[e2e][furnace]")
{
    const std::string output = render("white_furnace_teapot.nrscene", 4096, "white_furnace_teapot.exr");
    checkEnergyConservation(BitmapReader::read(output));
}
