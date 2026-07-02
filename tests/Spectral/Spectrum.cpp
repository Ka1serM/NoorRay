#include "Raytracing/RgbToSpectrum.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

extern const float sRGBToSpectrumTable_Scale[64];
extern const float sRGBToSpectrumTable_Data[3][64][64][64][3];

namespace
{
constexpr int IntegrationSamples = 4096;

glm::vec3 integrateIlluminant(const glm::vec3 rgb)
{
    glm::vec3 sum{};
    for (int i = 0; i < IntegrationSamples; ++i)
    {
        const SampledWavelengths wl = SampledWavelengths::sampleVisible(
            (i + 0.5f) / IntegrationSamples);
        const SampledSpectrum spectrum = rgbIlluminantToSpectrum(
            rgb, wl, sRGBToSpectrumTable_Scale,
            &sRGBToSpectrumTable_Data[0][0][0][0][0], NrD65);
        sum += xyzToLinearSRGB(spectrumToXYZ(
            spectrum, wl, NrCIE_X, NrCIE_Y, NrCIE_Z));
    }
    return sum / static_cast<float>(IntegrationSamples);
}

glm::vec3 integrateAlbedoUnderWhite(const glm::vec3 rgb)
{
    glm::vec3 sum{};
    for (int i = 0; i < IntegrationSamples; ++i)
    {
        const SampledWavelengths wl = SampledWavelengths::sampleVisible(
            (i + 0.5f) / IntegrationSamples);
        const SampledSpectrum albedo = rgbAlbedoToSpectrum(
            rgb, wl, sRGBToSpectrumTable_Scale,
            &sRGBToSpectrumTable_Data[0][0][0][0][0]);
        const SampledSpectrum white = rgbIlluminantToSpectrum(
            glm::vec3(1.0f), wl, sRGBToSpectrumTable_Scale,
            &sRGBToSpectrumTable_Data[0][0][0][0][0], NrD65);
        sum += xyzToLinearSRGB(spectrumToXYZ(
            albedo * white, wl, NrCIE_X, NrCIE_Y, NrCIE_Z));
    }
    return sum / static_cast<float>(IntegrationSamples);
}
}

TEST_CASE("visible wavelength samples and PDFs are finite and in range",
    "[spectral][wavelength]")
{
    for (int sample = 0; sample < 4096; ++sample)
    {
        const SampledWavelengths wl = SampledWavelengths::sampleVisible(
            (sample + 0.5f) / 4096.0f);
        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            CHECK(std::isfinite(wl.lambda[i]));
            CHECK(std::isfinite(wl.pdf[i]));
            CHECK(wl.lambda[i] >= NrLambdaMin);
            CHECK(wl.lambda[i] <= NrLambdaMax);
            CHECK(wl.pdf[i] > 0.0f);
        }
    }
}

TEST_CASE("visible wavelength estimator integrates a flat spectrum",
    "[spectral][wavelength]")
{
    glm::vec3 expected{};
    for (int i = 0; i < NrCIESamples; ++i)
        expected += glm::vec3(NrCIE_X[i], NrCIE_Y[i], NrCIE_Z[i]);
    expected /= NrCIE_Y_integral;

    glm::vec3 estimate{};
    for (int i = 0; i < IntegrationSamples; ++i)
    {
        const SampledWavelengths wl = SampledWavelengths::sampleVisible(
            (i + 0.5f) / IntegrationSamples);
        estimate += spectrumToXYZ(
            SampledSpectrum(1.0f), wl, NrCIE_X, NrCIE_Y, NrCIE_Z);
    }
    estimate /= static_cast<float>(IntegrationSamples);

    CHECK(estimate.x == Catch::Approx(expected.x).margin(2e-4f));
    CHECK(estimate.y == Catch::Approx(expected.y).margin(2e-4f));
    CHECK(estimate.z == Catch::Approx(expected.z).margin(2e-4f));
}

TEST_CASE("terminated hero wavelength estimator remains unbiased",
    "[spectral][wavelength][hero]")
{
    glm::vec3 expected{};
    for (int i = 0; i < NrCIESamples; ++i)
        expected += glm::vec3(NrCIE_X[i], NrCIE_Y[i], NrCIE_Z[i]);
    expected /= NrCIE_Y_integral;

    glm::vec3 estimate{};
    for (int i = 0; i < IntegrationSamples; ++i)
    {
        SampledWavelengths wl = SampledWavelengths::sampleVisible(
            (i + 0.5f) / IntegrationSamples);
        wl.terminateSecondary();
        SampledSpectrum heroOnly{};
        heroOnly[0] = 1.0f;
        estimate += spectrumToXYZ(
            heroOnly, wl, NrCIE_X, NrCIE_Y, NrCIE_Z);
    }
    estimate /= static_cast<float>(IntegrationSamples);

    CHECK(estimate.x == Catch::Approx(expected.x).margin(8e-4f));
    CHECK(estimate.y == Catch::Approx(expected.y).margin(8e-4f));
    CHECK(estimate.z == Catch::Approx(expected.z).margin(8e-4f));
}

TEST_CASE("sRGB illuminant upsampling round-trips through spectral integration",
    "[spectral][rgb]")
{
    for (const glm::vec3 rgb : {glm::vec3(1.0f), glm::vec3(0.8f, 0.2f, 0.1f),
             glm::vec3(0.05f, 0.4f, 0.9f), glm::vec3(3.0f, 0.5f, 1.5f)})
    {
        const glm::vec3 recovered = integrateIlluminant(rgb);
        INFO("rgb=" << rgb.x << "," << rgb.y << "," << rgb.z
             << " recovered=" << recovered.x << "," << recovered.y << "," << recovered.z);
        CHECK(recovered.x == Catch::Approx(rgb.x).margin(0.004f));
        CHECK(recovered.y == Catch::Approx(rgb.y).margin(0.004f));
        CHECK(recovered.z == Catch::Approx(rgb.z).margin(0.004f));
    }
}

TEST_CASE("sRGB albedo upsampling round-trips under a white D65 illuminant",
    "[spectral][rgb]")
{
    for (const glm::vec3 rgb : {glm::vec3(0.18f), glm::vec3(0.8f, 0.2f, 0.1f),
             glm::vec3(0.05f, 0.4f, 0.9f)})
    {
        const glm::vec3 recovered = integrateAlbedoUnderWhite(rgb);
        INFO("rgb=" << rgb.x << "," << rgb.y << "," << rgb.z
             << " recovered=" << recovered.x << "," << recovered.y << "," << recovered.z);
        CHECK(recovered.x == Catch::Approx(rgb.x).margin(0.004f));
        CHECK(recovered.y == Catch::Approx(rgb.y).margin(0.004f));
        CHECK(recovered.z == Catch::Approx(rgb.z).margin(0.004f));
    }
}
