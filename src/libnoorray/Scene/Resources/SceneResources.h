#pragma once

#include <vector>

#include "Backend/CUDA/rstd/Vector.h"
#include "Geometry/Mesh/Assets/GaussianAsset.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Scene/Resources/MaterialRegistry.h"
#include "Scene/Resources/ResourceRegistry.h"
#include "Scene/Resources/Texture.h"

// Mesh, material and Gaussian storage is shared with the CUDA kernels, so it
// uses the managed-memory vector; textures are host-side staging data that the
// renderer turns into CUDA texture objects of its own.
using MeshAssetRegistry =
    nr::ResourceRegistry<MeshAsset, nr::rstd::vector<MeshAsset>>;
using GaussianAssetRegistry =
    nr::ResourceRegistry<GaussianAsset, nr::rstd::vector<GaussianAsset>>;
using TextureRegistry = nr::ResourceRegistry<Texture, std::vector<Texture>>;

using MeshAssetRef = nr::ResourceRef<MeshAssetRegistry>;
using GaussianAssetRef = nr::ResourceRef<GaussianAssetRegistry>;
using TextureRef = nr::ResourceRef<TextureRegistry>;
