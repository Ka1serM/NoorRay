#pragma once

#include "Types.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

static const uint MaxLensSurfaces = 64;
static const uint MaxAsphereTerms = 8;

struct Medium
{
    float3 b;
    float3 c;
};

struct Surface
{
    float z;
    float radius;
    float apertureRadius;
    float conic;
    float asphere[MaxAsphereTerms];
    uint mediumAfter;
    uint isStop;
};

struct Lens
{
    Surface surfaces[MaxLensSurfaces];
    Medium media[MaxLensSurfaces + 1];
    uint surfaceCount;
    uint mediumCount;
    float rearPupilZ;
    float rearPupilRadius;
    float focalLengthMm;
    float sensorWidthMm;
    float sensorHeightMm;
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
