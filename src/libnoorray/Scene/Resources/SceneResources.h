#pragma once

#include <vector>

#include <vector>
#include "Geometry/Mesh/Assets/GaussianAsset.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Scene/Resources/MaterialRegistry.h"
#include "Scene/Resources/ResourceRegistry.h"
#include "Scene/Resources/Texture.h"

// Mesh, material and Gaussian storage is host-owned until the Vulkan renderer
// publishes immutable device buffers. Textures are host-side scene assets and are
// stored directly by Scene; the renderer uploads them as Vulkan images.
using MeshAssetRegistry =
    nr::ResourceRegistry<MeshAsset, std::vector<MeshAsset>>;
using GaussianAssetRegistry =
    nr::ResourceRegistry<GaussianAsset, std::vector<GaussianAsset>>;

using MeshAssetRef = nr::ResourceRef<MeshAssetRegistry>;
using GaussianAssetRef = nr::ResourceRef<GaussianAssetRegistry>;
