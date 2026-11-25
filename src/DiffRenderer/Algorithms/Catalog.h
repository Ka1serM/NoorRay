#pragma once

#include <array>
#include <span>
#include <string_view>

#include "DiffRenderer/Algorithms/GaussianSplatting/GaussianSplatting.h"
#ifdef NOORRAY_GAUSSIAN_TRACING
#include "DiffRenderer/Algorithms/GaussianTracing/GaussianTracing.h"
#endif
#include "DiffRenderer/Algorithms/PathTracing/PathTracing.h"
#include "DiffRenderer/Algorithms/Rasterization/Rasterization.h"
#include "DiffRenderer/Algorithms/SdfTracing/SdfTracing.h"
#include "DiffRenderer/Algorithms/VolumeRendering/VolumeRendering.h"
#include "DiffRenderer/Algorithms/XrayProjection/XrayProjection.h"

namespace DiffRenderer::Algorithms {

//! Owns one instance of every algorithm; an algorithm holds GPU resources only after its open.
class Catalog {
public:
    Catalog() = default;
    Catalog(const Catalog&) = delete;
    Catalog& operator=(const Catalog&) = delete;

    std::span<Algorithm* const> getAll() const { return all; }
    Algorithm* find(std::string_view id) const;

private:
    GaussianSplatting gaussianSplatting;
    Rasterization rasterization;
    VolumeRendering volumeRendering;
    XrayProjection xrayProjection;
    PathTracing pathTracing;
    SdfTracing sdfTracing;
#ifdef NOORRAY_GAUSSIAN_TRACING
    GaussianTracing gaussianTracing;
    std::array<Algorithm*, 7> all{&gaussianSplatting, &rasterization, &volumeRendering, &xrayProjection, &pathTracing, &sdfTracing, &gaussianTracing};
#else
    std::array<Algorithm*, 6> all{&gaussianSplatting, &rasterization, &volumeRendering, &xrayProjection, &pathTracing, &sdfTracing};
#endif
};

}
