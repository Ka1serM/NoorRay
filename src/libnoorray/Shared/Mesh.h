#pragma once

#include "Types.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

struct Instance
{
    float objectToWorld[12];
    uint meshIndex;
    uint reserved[3];
};

struct Vertex
{
    float3 position;
    float3 normal;
    float3 tangent;
    float tangentSign;
    float2 uv;
    // R | G << 8 | B << 16 | A << 24 (UNORM8).
    uint color;
};

struct Face
{
    int materialIndex;
};

struct Mesh
{
    GpuPtr(Vertex) vertices;
    GpuPtr(uint) indices;
    GpuPtr(Face) faces;
    GpuPtr(uint) materialIds;
    uint vertexCount;
    uint indexCount;
    uint faceCount;
    uint materialCount;
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
