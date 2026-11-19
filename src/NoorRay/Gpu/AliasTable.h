#pragma once

#include <span>
#include <vector>

#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

//! Vose's alias method. Weights that sum to zero give a uniform table. The table has at least one entry.
std::vector<Shaders::AliasEntry> buildAliasTable(std::span<const float> weights);

}
