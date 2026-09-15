#pragma once

#include "Mesh.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

struct Scene
{
    GpuPtr(GpuPtr(Mesh)) meshes;
    GpuPtr(Instance) instances;
    uint meshCount;
    uint instanceCount;
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
