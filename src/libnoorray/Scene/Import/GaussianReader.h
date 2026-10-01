#pragma once

#include <string>
#include <vector>

#include "Mesh/Assets/Gaussian.h"

// Reads a Gaussian splat file (.ply, .compressed.ply, .splat, .ksplat, .spz,
// .sog) into splats in NoorRay's coordinate space.
class GaussianReader
{
public:
    static std::vector<Gaussian> read(const std::string& path);
};
