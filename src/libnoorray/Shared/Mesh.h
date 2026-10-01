#pragma once

#include "Types.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

// A mesh instance as the hit stages see it. Every TLAS record of the instance
// carries its index as InstanceID(); the transform is the record's own.
struct Instance
{
    uint meshIndex;
    // Where this instance's entry starts in its material's parameters, for
    // materials that hold one entry per instance, such as Gaussian splats.
    uint materialEntry;
    // Scene material index per mesh material slot.
    GpuPtr(uint) materials;
    // Replaces the mesh's colors when set, in the same layout.
    GpuPtr(uint) colors;
    // Unreal's CustomPrimitiveData: floats materials read by index; those past
    // customDataCount read as zero.
    GpuPtr(float) customData;
    // Bit n is Unreal's lighting channel n; lights reach the instance only
    // through a channel they share.
    uint lightingChannels;
    uint customDataCount;
};

// A mesh section as the hit stages see it. Each section is one BLAS
// geometry, so a hit's triangle is sections[GeometryIndex()].firstTriangle +
// PrimitiveIndex().
struct SectionRecord
{
    uint firstTriangle;
    // Index into the instance's materials.
    uint slot;
    // Shadow rays pass the section when zero (MeshSection::castsShadow).
    uint castsShadow;
};

// Vertex streams in Unreal's layout (format/umap.fbs MeshLod), one entry per
// vertex each.
struct Mesh
{
    GpuPtr(float3) positions;
    // TangentX then TangentZ as SNORM16x4; TangentZ is the normal and
    // TangentZ.w the bitangent sign.
    GpuPtr(uint4) tangents;
    // uvCount channels per vertex: uvs[uvCount * vertex + channel].
    GpuPtr(float2) uvs;
    // Optional FColor per vertex: B | G << 8 | R << 16 | A << 24, linear
    // UNORM8. Null reads as white.
    GpuPtr(uint) colors;
    GpuPtr(uint) indices;
    GpuPtr(SectionRecord) sections;
    uint vertexCount;
    uint indexCount;
    uint sectionCount;
    uint uvCount;
    // Object-space bounds of the vertices, for materials that read them.
    float boundsMin[3];
    float boundsMax[3];
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
