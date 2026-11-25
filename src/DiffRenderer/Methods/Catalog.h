#pragma once

#include <array>
#include <span>
#include <string_view>

#include "DiffRenderer/Methods/GaussianSplatting/GaussianSplatting.h"
#ifdef NOORRAY_GAUSSIAN_TRACING
#include "DiffRenderer/Methods/GaussianTracing/GaussianTracing.h"
#endif
#include "DiffRenderer/Methods/Mesh/MeshRasterization.h"
#include "DiffRenderer/Methods/Method.h"
#include "DiffRenderer/Methods/SdfGrid/SdfGrid.h"
#include "DiffRenderer/Methods/VoxelGrid/VoxelGrid.h"

namespace DiffRenderer::Methods {

//! Owns one instance of every method; a method holds GPU resources only after its open.
class Catalog {
public:
    Catalog() = default;
    Catalog(const Catalog&) = delete;
    Catalog& operator=(const Catalog&) = delete;

    std::span<Method* const> getAll() const { return all; }
    Method* find(std::string_view id) const;
    const SdfGrid& getSdfGrid() const { return sdfGrid; }
    const MeshRasterization& getMeshRasterization() const { return meshRasterization; }

private:
    GaussianSplatting gaussianSplatting;
    SdfGrid sdfGrid;
    VoxelGrid voxelGrid;
    MeshRasterization meshRasterization;
#ifdef NOORRAY_GAUSSIAN_TRACING
    GaussianTracing gaussianTracing;
    std::array<Method*, 5> all{&gaussianSplatting, &gaussianTracing, &sdfGrid, &voxelGrid, &meshRasterization};
#else
    std::array<Method*, 4> all{&gaussianSplatting, &sdfGrid, &voxelGrid, &meshRasterization};
#endif
};

}
