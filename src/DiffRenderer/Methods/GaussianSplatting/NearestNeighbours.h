#pragma once

#include <span>
#include <vector>

#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Methods {

//! The mean squared distance of each point to its three nearest other points, as distCUDA2 of simple-knn computes it for the 3DGS start.
std::vector<float> getMeanSquaredNeighbourDistances(std::span<const NoorRHI::float3> points);

}
