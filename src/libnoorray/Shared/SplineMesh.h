#pragma once

#include "Types.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

// Unreal's USplineMeshComponent: the mesh bent along the Hermite curve from
// start to end, as its vertex factory bends it (SplineMeshCommon.ush). Roll is
// in radians.
struct SplineMesh
{
    float3 startPosition;
    float3 startTangent;
    float3 endPosition;
    float3 endTangent;
    float3 upDirection;
    float2 startScale;
    float2 endScale;
    float2 startOffset;
    float2 endOffset;
    float startRoll;
    float endRoll;
    // A vertex's distance along the spline, 0 at its start and 1 at its end,
    // is its forward coordinate * distanceScale + distanceOffset.
    float distanceScale;
    float distanceOffset;
    // The mesh axis along the spline, 0 to 2 for X to Z; the next two axes,
    // cyclically, are the slice's X and Y.
    uint forwardAxis;
    // Nonzero eases roll, scale and offset with smoothstep instead of linearly.
    uint smoothInterpRollScale;
};

// SplineMesh.slang's launch: bends the streams in place.
struct SplineMeshArguments
{
    GpuPtr(float3) positions;
    // Mesh::tangents' layout.
    GpuPtr(uint2) tangents;
    uint vertexCount;
    SplineMesh spline;
};

struct SplineMeshRoot
{
    GpuPtr(SplineMeshArguments) arguments;
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
