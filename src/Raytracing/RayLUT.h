#pragma once

#include "Shaders/Shared.h"

#include <filesystem>
#include <optional>
#include <vector>

struct RayLUT
{
    uint32_t rasterWidth = 0;
    uint32_t rasterHeight = 0;
    uint32_t stepSize = 1;
    uint32_t numWavelengths = 0;
    uint32_t lutWidth = 0;
    uint32_t lutHeight = 0;
    std::vector<float> wavelengths;
    std::vector<RayLutEntry> rays;

    RayLUT() = default;
    RayLUT(uint32_t rasterWidth, uint32_t rasterHeight, uint32_t stepSize);

    bool empty() const { return rays.empty() || numWavelengths == 0; }
    std::optional<RayLutEntry> lookupInterpolated(float x, float y, float wavelength) const;
};

class RayLUTFileReader
{
public:
    RayLUT read(const std::filesystem::path& path) const;
};

class RayLUTFileWriter
{
public:
    void write(const std::filesystem::path& path, const RayLUT& lut) const;
};
