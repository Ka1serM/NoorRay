#include "IO/BitmapReader.h"
#include "IO/BitmapWriter.h"
#include "RenderTestFixture.h"

#include <catch2/catch_test_macros.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace
{

std::string shellQuote(const std::string& value)
{
    std::string result = "'";
    for (const char character : value) {
        if (character == '\'') result += "'\\''";
        else result += character;
    }
    return result + "'";
}

}

class MaterialXFidelityTest : public RenderTestFixture {};

TEST_CASE_METHOD(MaterialXFidelityTest,
    "MaterialX validator materials match the independent MaterialXView reference",
    "[e2e][fidelity][materialx]")
{
    struct Fixture { const char* name; const char* material; const char* scene; };
    constexpr Fixture fixtures[] = {
        {"open_pbr_default", "open_pbr_default.mtlx", "materialx_validator_open_pbr.nrscene"},
        {"standard_surface_default", "standard_surface_default.mtlx", "materialx_validator_standard_surface.nrscene"},
        {"gltf_pbr_default", "gltf_pbr_default.mtlx", "materialx_validator_gltf_pbr.nrscene"},
    };
    const std::filesystem::path outputDirectory =
        std::filesystem::temp_directory_path() / "noorray-materialx-fidelity";
    std::filesystem::create_directories(outputDirectory);
    REQUIRE(std::filesystem::is_regular_file(MATERIALXVIEW_EXECUTABLE));

    for (const Fixture& fixture : fixtures) {
        SECTION(fixture.name) {
            // MaterialXView captures PNG. BitmapReader's floating-point stb
            // path already converts LDR values to linear for comparison with
            // NoorRay's linear EXR.
            const auto reference = outputDirectory / (std::string(fixture.name) + "-materialxview.png");
            const auto actual = outputDirectory / (std::string(fixture.name) + "-noorray.exr");
            const auto material = std::filesystem::path(TEST_ASSET_DIR) / "materialx/validator" / fixture.material;
            const auto mesh = std::filesystem::path(MATERIALX_RESOURCE_DIR) / "Geometry/sphere.obj";
            const auto envRad = std::filesystem::path(MATERIALX_RESOURCE_DIR) / "Lights/san_giuseppe_bridge.hdr";
            std::ostringstream viewerCommand;
            viewerCommand << shellQuote(MATERIALXVIEW_EXECUTABLE)
                          << " --material " << shellQuote(material.string())
                          << " --mesh " << shellQuote(mesh.string())
                          << " --envRad " << shellQuote(envRad.string())
                          << " --path " << shellQuote(MATERIALX_SOURCE_DIR)
                          << " --screenWidth 256 --screenHeight 256"
                          << " --screenColor 0,0,0 --drawEnvironment true"
                          << " --enableTurntable false --shadowMap false"
                          << " --captureFilename " << shellQuote(reference.string());
            REQUIRE(std::system(viewerCommand.str().c_str()) == 0);
            REQUIRE(std::filesystem::is_regular_file(reference));

            const Bitmap capturedReference = BitmapReader::read(reference.string());
            const std::string rendered = render(fixture.scene, 64, actual.string().c_str(),
                capturedReference.width(), capturedReference.height());
            const Bitmap actualBitmap = BitmapReader::read(rendered);
            REQUIRE(capturedReference.width() == actualBitmap.width());
            REQUIRE(capturedReference.height() == actualBitmap.height());
            const Bitmap& expectedBitmap = capturedReference;

            double squaredError = 0.0;
            double absoluteError = 0.0;
            std::vector<glm::vec4> differencePixels;
            differencePixels.reserve(expectedBitmap.pixels().size());
            for (size_t index = 0; index < expectedBitmap.pixels().size(); ++index) {
                glm::vec4 differencePixel(0.0f);
                for (int channel = 0; channel < 3; ++channel) {
                    const float difference = actualBitmap.pixels()[index][channel]
                        - expectedBitmap.pixels()[index][channel];
                    squaredError += static_cast<double>(difference) * difference;
                    absoluteError += std::abs(difference);
                    differencePixel[channel] = std::min(std::abs(difference) * 8.0f, 1.0f);
                }
                differencePixel.a = 1.0f;
                differencePixels.push_back(differencePixel);
            }
            const double samples = static_cast<double>(expectedBitmap.pixels().size()) * 3.0;
            const double mse = squaredError / samples;
            const double psnr = mse == 0.0 ? INFINITY : 10.0 * std::log10(1.0 / mse);
            INFO("MaterialXView reference: " << reference);
            INFO("NoorRay render: " << actual);
            INFO("MAE: " << absoluteError / samples << ", PSNR: " << psnr << " dB");
            const auto diffPath = outputDirectory / (std::string(fixture.name) + "-diff.png");
            REQUIRE(BitmapWriter::write(diffPath.string(),
                Bitmap(expectedBitmap.width(), expectedBitmap.height(), std::move(differencePixels))));
            CHECK(absoluteError / samples < 0.25);
        }
    }
}
