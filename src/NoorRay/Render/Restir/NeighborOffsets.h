#pragma once

#include <cstdint>
#include <vector>

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace NoorRay::Render {

//! The spatial neighbors of RTXDI: a power of two, so that the shaders wrap the index with neighborCount - 1.
constexpr std::uint32_t neighborOffsetCount = 8192;

//! RTXDI's offsets of the spatial neighbors in the unit disk, as floats.
std::vector<NoorRHI::float2> makeNeighborOffsets();

}
