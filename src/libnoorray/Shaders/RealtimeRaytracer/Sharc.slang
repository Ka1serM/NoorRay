#pragma once

#include "RealtimeArgs.slang"

// Includes NVIDIA's SHaRC radiance cache (external/SHARC, HLSL headers) with
// NoorRay's resources. The SHARC headers address structured buffers through
// two macros; here they become device pointers from RadianceCacheArgs, so the
// cache needs no descriptor bindings. SHARC_UPDATE / SHARC_QUERY select the
// stage and are passed on the slangc command line.

#ifndef SHARC_UPDATE
#define SHARC_UPDATE 0
#endif
#ifndef SHARC_QUERY
#define SHARC_QUERY 0
#endif
#define SHARC_ENABLE_GLSL 0
#define SHARC_ENABLE_RESPONSIVE_LIGHTING 0
#define RW_STRUCTURED_BUFFER(name, type) type* name
#define BUFFER_AT_OFFSET(name, offset) name[offset]

#include "SharcCommon.h"

// Grid levels grow with log2 of the camera distance, so a voxel spans about
// distance / sceneScale in any unit system.
SharcParameters makeSharcParameters()
{
    RealtimeArgs* args = realtimeArgs();
    SharcParameters parameters;
    parameters.hashGridParameters.cameraPosition = args.view.cameraPosition;
    parameters.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    parameters.hashGridParameters.sceneScale = args.radianceCache.sceneScale;
    parameters.hashGridParameters.levelBias = float(SHARC_GRID_LEVEL_BIAS);
    parameters.hashGridData.capacity = args.radianceCache.capacity;
    parameters.hashGridData.hashEntriesBuffer = args.radianceCache.hashEntries;
    parameters.radianceScale = args.radianceCache.radianceScale;
    parameters.accumulationBuffer = (SharcAccumulationData*)args.radianceCache.accumulation;
    parameters.resolvedBuffer = (SharcPackedData*)args.radianceCache.resolved;
    return parameters;
}

bool sharcEnabled()
{
    return realtimeArgs()->radianceCache.capacity != 0u;
}
