#include "NoorRay/Spectral/SpectralTables.h"

#include <algorithm>
#include <array>
#include <iterator>

#include "NoorRay/Spectral/BasisTable.h"
#include "NoorRay/Spectral/Blackbody.h"
#include "NoorRay/Spectral/CieTables.h"
#include "NoorRay/Spectral/Conductors.h"
#include "NoorRay/Spectral/WavelengthSampler.h"

namespace NoorRay::Spectral {

namespace {

using Matrix = std::array<std::array<double, 3>, 3>;

Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix result{};
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            for (int k = 0; k < 3; ++k)
                result[row][column] += a[row][k] * b[k][column];
    return result;
}

Matrix invert(const Matrix& m) {
    const double determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                               + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    Matrix result{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            const int r0 = (column + 1) % 3, r1 = (column + 2) % 3, c0 = (row + 1) % 3, c1 = (row + 2) % 3;
            result[row][column] = (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]) / determinant;
        }
    }
    return result;
}

double sampleTable(const std::array<float, cieSampleCount>& table, const double lambda) {
    const double position = std::clamp(lambda - cieLambdaMin, 0.0, static_cast<double>(cieSampleCount - 1));
    const std::size_t index = std::min(static_cast<std::size_t>(position), cieSampleCount - 2);
    const double weight = position - static_cast<double>(index);
    return table[index] * (1.0 - weight) + table[index + 1] * weight;
}

struct FilmResponse {
    Matrix xyzToRgb{};
    std::array<double, 3> white{};
};

FilmResponse computeFilmResponse() {
    const Matrix xyzToSrgb{{{3.2404542, -1.5371385, -0.4985314}, {-0.9692660, 1.8760108, 0.0415560}, {0.0556434, -0.2040259, 1.0572252}}};
    const Matrix bradford{{{0.8951, 0.2664, -0.1614}, {-0.7502, 1.7135, 0.0367}, {0.0389, -0.0685, 1.0296}}};
    const std::array<double, 3> d65{0.95047, 1.0, 1.08883};
    Matrix scale{};
    for (int row = 0; row < 3; ++row) {
        const double coneD65 = bradford[row][0] * d65[0] + bradford[row][1] * d65[1] + bradford[row][2] * d65[2];
        const double coneE = bradford[row][0] + bradford[row][1] + bradford[row][2];
        scale[row][row] = coneD65 / coneE;
    }
    FilmResponse response;
    response.xyzToRgb = multiply(xyzToSrgb, multiply(invert(bradford), multiply(scale, bradford)));
    for (std::size_t i = 0; i < cieSampleCount; ++i) {
        const double xyz[3] = {cieX[i], cieY[i], cieZ[i]};
        for (int row = 0; row < 3; ++row)
            response.white[row] += response.xyzToRgb[row][0] * xyz[0] + response.xyzToRgb[row][1] * xyz[1] + response.xyzToRgb[row][2] * xyz[2];
    }
    return response;
}

const FilmResponse& filmResponse() {
    static const FilmResponse response = computeFilmResponse();
    return response;
}

}

Core::float3 SpectralTables::basis(const float lambda) {
    const float position = std::clamp((lambda - basisLambdaMin) / basisLambdaStep, 0.0f, static_cast<float>(basisSampleCount - 1));
    const std::size_t index = std::min(static_cast<std::size_t>(position), basisSampleCount - 2);
    const float weight = position - static_cast<float>(index);
    const auto lerp = [&](const std::array<float, basisSampleCount>& table) { return table[index] * (1.0f - weight) + table[index + 1] * weight; };
    return {lerp(basisRed), lerp(basisGreen), lerp(basisBlue)};
}

Core::float3 SpectralTables::film(const float lambda) {
    const FilmResponse& response = filmResponse();
    const double xyz[3] = {sampleTable(cieX, lambda), sampleTable(cieY, lambda), sampleTable(cieZ, lambda)};
    float rgb[3];
    for (int row = 0; row < 3; ++row)
        rgb[row] = static_cast<float>((response.xyzToRgb[row][0] * xyz[0] + response.xyzToRgb[row][1] * xyz[1] + response.xyzToRgb[row][2] * xyz[2])
                                      / response.white[row]);
    return {rgb[0], rgb[1], rgb[2]};
}

std::vector<Shaders::SpectralEntry> SpectralTables::buildEntries() {
    std::vector<Shaders::SpectralEntry> table(Shaders::spectralTableSize);
    for (std::uint32_t i = 0; i < Shaders::spectralTableSize; ++i) {
        const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(Shaders::spectralTableSize);
        const float lambda = sampleVisibleWavelength(u);
        const float pdf = visibleWavelengthPdf(lambda);
        const Core::float3 b = basis(lambda);
        const Core::float3 f = film(lambda) / pdf;
        table[i] = {{b.x, b.y, b.z, lambda}, {f.x, f.y, f.z, pdf}};
    }
    return table;
}

std::vector<Shaders::ConductorSample> SpectralTables::buildConductors() {
    std::vector<Shaders::ConductorSample> samples;
    samples.reserve(conductorCount * Shaders::conductorSampleCount);
    for (std::size_t conductor = 0; conductor < conductorCount; ++conductor)
        std::ranges::copy(getConductorSamples(static_cast<Conductor>(conductor)), std::back_inserter(samples));
    return samples;
}

std::expected<void, Core::Error> SpectralTables::open(NoorRHI::Device& device) {
    const std::vector<Shaders::SpectralEntry> entryValues = buildEntries();
    const std::vector<Shaders::ConductorSample> conductorValues = buildConductors();
    if (std::expected<void, NoorRHI::Error> opened = entries.open(device, entryValues.size()); !opened)
        return Core::fail(std::move(opened.error()));
    if (std::expected<void, NoorRHI::Error> uploaded = entries.upload(entryValues); !uploaded)
        return Core::fail(std::move(uploaded.error()));
    if (std::expected<void, NoorRHI::Error> opened = conductors.open(device, conductorValues.size()); !opened)
        return Core::fail(std::move(opened.error()));
    return Core::check(conductors.upload(conductorValues));
}

Core::float3 blackbodyColor(const float kelvin) {
    Core::float3 rgb;
    for (float lambda = cieLambdaMin; lambda <= cieLambdaMax; lambda += 1.0f)
        rgb += SpectralTables::film(lambda) * planck(lambda, kelvin);
    const float y = Core::luminance(rgb);
    return y > 0.0f ? rgb / y : Core::float3{};
}

}
