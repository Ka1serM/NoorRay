#pragma once

#include <cstdint>
#include <vector>

#include "CUDA/Unique/Texture.h"
#include "CUDA/rstd/Vector.h"
#include "Raytracing/Gpu/SceneData.h"
#include "Scene/GpuInstance.h"
#include "Scene/SceneResources.h"

struct GpuSceneCache
{
    GpuSceneData data{};
    nr::rstd::vector<GpuInstance> instances;
    nr::rstd::vector<Material> materials;
    nr::rstd::vector<nr::cuda::UniqueTexture> textures;
    // Generation-checked CPU registry handles mirrored by each CUDA slot.
    // They let texture membership changes update only the affected arrays.
    std::vector<TextureHandle> textureHandles;
    uint64_t textureRegistryRevision{};

    void clearSceneResources()
    {
        textures.clear();
        textureHandles.clear();
        textureRegistryRevision = 0;
        instances.clear();
        materials.clear();
        data = {};
    }
};
