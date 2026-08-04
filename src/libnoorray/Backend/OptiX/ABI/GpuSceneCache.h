#pragma once

#include <cstdint>
#include <vector>

#include "Backend/CUDA/Unique/Texture.h"
#include "Backend/CUDA/rstd/Vector.h"
#include "Backend/OptiX/ABI/SceneData.h"
#include "Scene/GpuInstance.h"
#include "Scene/Resources/SceneResources.h"

struct GpuSceneCache
{
    GpuSceneData data{};
    nr::rstd::vector<GpuInstance> instances;
    nr::rstd::vector<Material> materials;
    nr::rstd::vector<nr::cuda::UniqueTexture> textures;
    // Generation-checked CPU registry handles mirrored by each CUDA slot.
    // They let texture membership changes update only the affected arrays.
    std::vector<TextureHandle> textureHandles;
    uint64_t textureRevision{};

    void clearSceneResources()
    {
        textures.clear();
        textureHandles.clear();
        textureRevision = 0;
        instances.clear();
        materials.clear();
        data = {};
    }
};
