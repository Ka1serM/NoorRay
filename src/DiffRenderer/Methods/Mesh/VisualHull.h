#pragma once

#include <optional>

#include "DiffRenderer/Methods/Mesh/SurfaceMesh.h"
#include "DiffRenderer/Scenes/Dataset.h"

namespace DiffRenderer::Methods {

//! The surface of the points every train view sees inside its mask, by marching tetrahedra over the box of the points; empty without masks.
std::optional<SurfaceMesh> carveVisualHull(const Scenes::Dataset& dataset, int resolution);

}
