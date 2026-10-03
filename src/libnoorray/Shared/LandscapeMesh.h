#pragma once

#include "Types.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

struct LandscapeMeshArguments
{
    GpuPtr(float3) positions;
    uint vertexCount;
};

struct LandscapeMeshRoot
{
    GpuPtr(LandscapeMeshArguments) arguments;
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
