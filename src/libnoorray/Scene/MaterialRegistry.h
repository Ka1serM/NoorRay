#pragma once

#include "CUDA/rstd/Vector.h"
#include "Scene/ResourceRegistry.h"
#include "Shading/Material.h"

// Material is a GPU-side struct, so reclaiming a slot just resets it to the
// default instead of adding a host-only teardown method to its layout. The
// overload lives beside Material so the registry finds it by argument-dependent
// lookup.
inline void releaseResource(Material& material)
{
    material = Material{};
}

using MaterialRegistry = nr::ResourceRegistry<Material, nr::rstd::vector<Material>>;
using MaterialRef = nr::ResourceRef<MaterialRegistry>;
