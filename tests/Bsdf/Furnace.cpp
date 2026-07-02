#include <catch2/catch_test_macros.hpp>

#include "IO/BitmapReader.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <string>

static std::string render(const char* scene, int spp, const char* output)
{
    const std::string path = output;
    const std::string command = std::string("\"") + NOORRAY_EXECUTABLE + "\" \""
        + TEST_SCENE_DIR "/" + scene + "\" " + std::to_string(spp)
        + " \"" + path + "\"";
    INFO("Command: " << command);
    REQUIRE(std::system(command.c_str()) == 0);
    return path;
}

static void checkEnergyConservation(const Bitmap& bitmap,
    float biasTolerance = 0.001f, float rmseTolerance = 0.004f)
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
        CHECK(std::abs(relativeBias) <= biasTolerance);
        CHECK(relativeRmse <= rmseTolerance);
    }
}

TEST_CASE("furnace metallic 1 (mirror) conserves energy", "[bsdf][furnace]")
{
    const std::string output = render("furnace_metallic_1.json", 1024,
        "furnace_metallic_1.exr");
    checkEnergyConservation(BitmapReader::read(output), 0.001f, 0.006f);
}

TEST_CASE("furnace metallic 1 roughness 1 (rough metal) conserves energy", "[bsdf][furnace]")
{
    const std::string output = render("furnace_metallic_1_roughness_1.json", 1024,
        "furnace_metallic_1_roughness_1.exr");
    checkEnergyConservation(BitmapReader::read(output), 0.001f, 0.006f);
}

TEST_CASE("furnace roughness 0 (diffuse) conserves energy", "[bsdf][furnace]")
{
    const std::string output = render("furnace_roughness_0.json", 1024,
        "furnace_roughness_0.exr");
    checkEnergyConservation(BitmapReader::read(output), 0.001f, 0.006f);
}

TEST_CASE("furnace roughness 1 (rough diffuse) conserves energy", "[bsdf][furnace]")
{
    const std::string output = render("furnace_roughness_1.json", 1024,
        "furnace_roughness_1.exr");
    checkEnergyConservation(BitmapReader::read(output), 0.001f, 0.006f);
}

TEST_CASE("furnace specular 1 conserves energy", "[bsdf][furnace]")
{
    const std::string output = render("furnace_specular_1.json", 1024,
        "furnace_specular_1.exr");
    checkEnergyConservation(BitmapReader::read(output), 0.001f, 0.006f);
}

TEST_CASE("furnace transmission 1 (glass) conserves energy", "[bsdf][furnace]")
{
    const std::string output = render("furnace_transmission_1.json", 4096,
        "furnace_transmission_1.exr");
    checkEnergyConservation(BitmapReader::read(output));
}
