#pragma once

#include "CUDA/Unique/Texture.h"
#include "CUDA/rstd/Vector.h"
#include "Raytracing/Gpu/SceneData.h"
#include "Scene/GpuInstance.h"

struct GpuSceneCache
{
    GpuSceneData data{};
    nr::rstd::vector<GpuInstance> instances;
    nr::rstd::vector<nr::cuda::UniqueTexture> textures;

    void clearSceneResources()
    {
        textures.clear();
        instances.clear();
        data = {};
    }
};
