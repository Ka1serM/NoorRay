#pragma once

#include <string>

#include "Mesh/Assets/Gaussian.h"

// Reads a Gaussian splat file (.ply, .compressed.ply, .splat, .ksplat, .spz,
// .sog) into a GaussianAsset in NoorRay's coordinate space.
class GaussianReader
{
public:
    static GaussianAsset read(Scene& scene, const std::string& name, const std::string& path);
};
