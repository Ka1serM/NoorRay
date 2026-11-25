#pragma once

#include <expected>
#include <filesystem>
#include <span>
#include <string>

#include "DiffRenderer/Methods/SdfGrid/Shaders/SurfaceRendering.h"

namespace DiffRenderer::Methods {

//! Extracts the zero level set of the node values by marching tetrahedra (six per cell) and writes it as a binary little-endian PLY.
[[nodiscard]] std::expected<void, std::string> writeIsoSurface(std::span<const float> values, const SurfaceRendering::Grid& grid,
                                                               const std::filesystem::path& path);

}
