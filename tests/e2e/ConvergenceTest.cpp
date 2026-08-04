#include "RenderTestFixture.h"

#include "IO/BitmapReader.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

class ConvergenceTest : public RenderTestFixture {};

namespace
{

double rmse(const Bitmap& image, const Bitmap& reference)
{
    REQUIRE(image.width() == reference.width());
    REQUIRE(image.height() == reference.height());

    double squaredError = 0.0;
    for (size_t pixel = 0; pixel < image.pixels().size(); ++pixel) {
        for (int channel = 0; channel < 3; ++channel) {
            const double error = static_cast<double>(image.pixels()[pixel][channel])
                - reference.pixels()[pixel][channel];
            squaredError += error * error;
        }
    }
    return std::sqrt(squaredError
        / (static_cast<double>(image.pixels().size()) * 3.0));
}

}

TEST_CASE_METHOD(ConvergenceTest,
    "flat material error decreases as samples increase", "[e2e][convergence]")
{
    const std::filesystem::path outputDirectory =
        std::filesystem::temp_directory_path() / "noorray-convergence";
    std::filesystem::create_directories(outputDirectory);

    constexpr uint32_t resolution = 96;
    constexpr const char* scene = "materialx_flat_red.nrscene";
    constexpr int referenceSamples = 256;
    constexpr int measuredSamples[] = {1, 4, 16, 64};

    const Bitmap reference = BitmapReader::read(render(
        scene, referenceSamples,
        (outputDirectory / "reference.exr").string().c_str(), resolution, resolution,
        0x4f1bbcddu));

    std::vector<double> errors;
    for (const int samples : measuredSamples) {
        const Bitmap image = BitmapReader::read(render(
            scene, samples,
            (outputDirectory / (std::to_string(samples) + "-spp.exr")).string().c_str(),
            resolution, resolution));
        const double error = rmse(image, reference);
        INFO("samples=" << samples << " RMSE=" << error);
        errors.push_back(error);
    }

    std::cout << "Convergence RMSE (reference=" << referenceSamples << " spp):";
    for (size_t index = 0; index < errors.size(); ++index)
        std::cout << ' ' << measuredSamples[index] << "=" << errors[index];
    std::cout << '\n';

    // The reference is independent, so these checks measure actual estimator
    // error. Keep generous margins for GPU/driver variation while catching a
    // sampler that stops covering new pixel samples.
    CHECK(errors[1] < errors.front() * 0.75);
    CHECK(errors[2] < errors.front() * 0.40);
    CHECK(errors.back() < errors.front() * 0.25);
}
