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

// A mesh section as the hit stages see it. Each section is one BLAS
// geometry, so a hit's triangle is sections[GeometryIndex()].firstTriangle +
// PrimitiveIndex().
struct SectionRecord
{
    uint firstTriangle;
    // Scene material index.
    uint material;
};

struct Mesh
{
    GpuPtr(Vertex) vertices;
    GpuPtr(uint) indices;
    GpuPtr(SectionRecord) sections;
    uint vertexCount;
    uint indexCount;
    uint sectionCount;
    uint padding;
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
