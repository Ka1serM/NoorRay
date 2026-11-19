#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

#include "NoorRay/Shaders/Spectral/SpectralRecords.h"

namespace NoorRay::Spectral {

enum class Conductor : std::uint8_t { Gold, Silver, Copper, Aluminium, Iron, Chromium };

inline constexpr std::size_t conductorCount = 6;
inline constexpr std::array<std::string_view, conductorCount> conductorNames{"Au", "Ag", "Cu", "Al", "Fe", "Cr"};

//! Measured complex IOR of the metals, resampled from refractiveindex.info (Johnson and Christy for Au, Ag, Cu; Rakic for Al; Johnson and Christy for Fe, Cr).
std::span<const Shaders::ConductorSample, Shaders::conductorSampleCount> getConductorSamples(Conductor conductor);

}
