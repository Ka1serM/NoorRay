#include "RayLUT.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <glm/geometric.hpp>

namespace {
struct RayLUTFileHeader
{
    char magic[8];
    uint32_t version;
    uint32_t rasterWidth;
    uint32_t rasterHeight;
    uint32_t stepSize;
    uint32_t numWavelengths;
};

constexpr char RayLUTMagic[8] = {'R', 'A', 'Y', 'L', 'U', 'T', '\0', '\0'};
constexpr uint32_t RayLUTVersion = 2u;

bool isFinite(const glm::vec3& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool isValid(const RayLutEntry& ray)
{
    return ray.originValid != 0.0f && isFinite(ray.origin) && isFinite(ray.direction);
}

glm::vec3 lerp(const glm::vec3& a, const glm::vec3& b, float t)
{
    return a * (1.0f - t) + b * t;
}
}

RayLUT::RayLUT(uint32_t rasterWidth, uint32_t rasterHeight, uint32_t stepSize)
    : rasterWidth(rasterWidth), rasterHeight(rasterHeight), stepSize(stepSize)
{
    if (rasterWidth == 0 || rasterHeight == 0)
        throw std::runtime_error("RayLUT raster resolution must be positive.");
    if (stepSize == 0)
        throw std::runtime_error("RayLUT step size must be positive.");

    lutWidth = (rasterWidth - 1u + stepSize - 1u) / stepSize + 1u;
    lutHeight = (rasterHeight - 1u + stepSize - 1u) / stepSize + 1u;
}

std::optional<RayLutEntry> RayLUT::lookupInterpolated(float x, float y, float wavelength) const
{
    if (empty() || lutWidth == 0 || lutHeight == 0)
        return std::nullopt;

    x = std::clamp(x, 0.0f, static_cast<float>(rasterWidth - 1u));
    y = std::clamp(y, 0.0f, static_cast<float>(rasterHeight - 1u));

    uint32_t nearestWavelengthIndex = 0;
    float nearestWavelengthDistance = std::abs(wavelength - wavelengths[0]);
    for (uint32_t i = 1; i < numWavelengths; ++i) {
        const float distance = std::abs(wavelength - wavelengths[i]);
        if (distance < nearestWavelengthDistance) {
            nearestWavelengthDistance = distance;
            nearestWavelengthIndex = i;
        }
    }

    const uint32_t x0 = static_cast<uint32_t>(x) / stepSize;
    const uint32_t y0 = static_cast<uint32_t>(y) / stepSize;
    const uint32_t x1 = std::min(x0 + 1u, lutWidth - 1u);
    const uint32_t y1 = std::min(y0 + 1u, lutHeight - 1u);
    const size_t bandOffset = static_cast<size_t>(nearestWavelengthIndex) * lutWidth * lutHeight;

    const RayLutEntry& r00 = rays[bandOffset + x0 + y0 * lutWidth];
    const RayLutEntry& r10 = rays[bandOffset + x1 + y0 * lutWidth];
    const RayLutEntry& r01 = rays[bandOffset + x0 + y1 * lutWidth];
    const RayLutEntry& r11 = rays[bandOffset + x1 + y1 * lutWidth];
    if (!isValid(r00) || !isValid(r10) || !isValid(r01) || !isValid(r11))
        return std::nullopt;

    const float px0 = static_cast<float>(std::min(x0 * stepSize, rasterWidth - 1u));
    const float py0 = static_cast<float>(std::min(y0 * stepSize, rasterHeight - 1u));
    const float px1 = static_cast<float>(std::min(x1 * stepSize, rasterWidth - 1u));
    const float py1 = static_cast<float>(std::min(y1 * stepSize, rasterHeight - 1u));
    const float tx = px1 == px0 ? 0.0f : (x - px0) / (px1 - px0);
    const float ty = py1 == py0 ? 0.0f : (y - py0) / (py1 - py0);

    RayLutEntry result{};
    result.origin = lerp(lerp(r00.origin, r10.origin, tx), lerp(r01.origin, r11.origin, tx), ty);
    result.direction = glm::normalize(lerp(lerp(r00.direction, r10.direction, tx), lerp(r01.direction, r11.direction, tx), ty));
    result.originValid = 1.0f;
    return result;
}

RayLUT RayLUTFileReader::read(const std::filesystem::path& path) const
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Could not open RayLUT file for reading.");

    RayLUTFileHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in)
        throw std::runtime_error("Failed while reading RayLUT file header.");
    if (std::memcmp(header.magic, RayLUTMagic, sizeof(RayLUTMagic)) != 0)
        throw std::runtime_error("Invalid RayLUT file magic.");
    if (header.version != RayLUTVersion)
        throw std::runtime_error("Unsupported RayLUT file version.");
    if (header.numWavelengths == 0)
        throw std::runtime_error("RayLUT file must contain at least one wavelength.");

    RayLUT lut(header.rasterWidth, header.rasterHeight, header.stepSize);
    lut.numWavelengths = header.numWavelengths;
    lut.wavelengths.resize(lut.numWavelengths);
    lut.rays.resize(static_cast<size_t>(lut.numWavelengths) * lut.lutWidth * lut.lutHeight);

    for (uint32_t w = 0; w < lut.numWavelengths; ++w)
        in.read(reinterpret_cast<char*>(&lut.wavelengths[w]), sizeof(float));
    if (!in)
        throw std::runtime_error("Failed while reading RayLUT file wavelengths.");

    for (RayLutEntry& ray : lut.rays) {
        in.read(reinterpret_cast<char*>(&ray.origin), sizeof(glm::vec3));
        in.read(reinterpret_cast<char*>(&ray.direction), sizeof(glm::vec3));
        ray.originValid = isFinite(ray.origin) && isFinite(ray.direction) ? 1.0f : 0.0f;
    }
    if (!in)
        throw std::runtime_error("Failed while reading RayLUT file rays.");

    return lut;
}

void RayLUTFileWriter::write(const std::filesystem::path& path, const RayLUT& lut) const
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("Could not open RayLUT file for writing.");

    RayLUTFileHeader header{};
    std::memcpy(header.magic, RayLUTMagic, sizeof(RayLUTMagic));
    header.version = RayLUTVersion;
    header.rasterWidth = lut.rasterWidth;
    header.rasterHeight = lut.rasterHeight;
    header.stepSize = lut.stepSize;
    header.numWavelengths = lut.numWavelengths;
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for (float wavelength : lut.wavelengths)
        out.write(reinterpret_cast<const char*>(&wavelength), sizeof(float));

    for (const RayLutEntry& ray : lut.rays) {
        out.write(reinterpret_cast<const char*>(&ray.origin), sizeof(glm::vec3));
        out.write(reinterpret_cast<const char*>(&ray.direction), sizeof(glm::vec3));
    }

    if (!out)
        throw std::runtime_error("Failed while writing RayLUT file.");
}
