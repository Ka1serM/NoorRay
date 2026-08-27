#pragma once

#include <cstdint>
#include <vector>

namespace nr::vulkan
{

// Packed unorm16 energy-compensation LUTs, in the element order
// EnergyLut.slang indexes: ggx directional, ggx average, dielectric
// directional, dielectric average, glass directional, glass average.
std::vector<std::uint16_t> packEnergyLutTables();

// CIE X/Y/Z colour matching functions, CIE D65, and the Jakob-Hanika 64^3
// sRGB-to-spectrum scale/coefficient tables, matching Spectrum.slang and
// RgbToSpectrum.slang. Keeping them contiguous costs one immutable heap slot.
std::vector<float> packSpectralTables();

} // namespace nr::vulkan
