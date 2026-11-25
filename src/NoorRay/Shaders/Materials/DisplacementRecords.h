#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/Shared.h"
//! Slang has the material records from the SceneRecords module.
#if !defined(__SLANG__)
#include "NoorRay/Shaders/Materials/MaterialRecords.h"
#endif

NR_NAMESPACE_BEGIN

static const uint displacementGroupSize = 128u;
//! Blocks per dispatch row, below the limit of 65535 workgroups per dimension.
static const uint displacementRowGroups = 32768u;

//! Where a displaced stream goes: nowhere, one value per vertex, or one per triangle corner.
static const uint displacementStreamNone = 0u;
static const uint displacementStreamVertex = 1u;
static const uint displacementStreamCorner = 2u;

//! One displaced mesh. Vertices and faces are those of the mesh and its MeshTopology; positions are three floats each.
struct DisplacementRecord {
    Ptr<uint2> instructions;
    Ptr<float4> registers;
    //! Attribute after attribute, one value per vertex.
    Ptr<float4> attributes;
    Ptr<float> restPositions;
    //! The angle-weighted vertex normals of the rest mesh, along which Cycles displaces.
    Ptr<float> restNormals;
    Ptr<float> faceNormals;
    Ptr<float> vertexNormals;
    Ptr<uint> cornerVertices;
    Ptr<uint> faceOffsets;
    //! The corners of each vertex: corners[cornerOffsets[v]..cornerOffsets[v + 1]] hold (face, corner).
    Ptr<uint> cornerOffsets;
    Ptr<uint2> corners;
    Ptr<uint> restTangents;
    //! The streams of the mesh, which the displaced vertices overwrite.
    Ptr<float> positions;
    Ptr<uint> indices;
    Ptr<uint> normals;
    Ptr<uint> tangents;
    //! The rest bounds give the generated texture space.
    float4 boundsMin;
    float4 boundsMax;
    uint instructionCount;
    uint registerCount;
    uint displacement;
    uint normalStream;
    uint tangentStream;
    uint vertexCount;
    uint faceCount;
    uint cornerCount;
};

struct DisplacementArguments {
    Ptr<DisplacementRecord> record;
};

//! The rest vertices that the displacement graph of one material moves, each as a part of the first triangle with a displacing material, as in Cycles.
struct DisplacementGraphRecord {
    Ptr<DisplacementRecord> mesh;
    Ptr<float4> parameters;
    Ptr<MaterialTextureRecord> textures;
    //! The vertex and the triangle corner of each point.
    Ptr<uint2> points;
    //! The shading normal and the triangle normal of each point, three floats each.
    Ptr<float> normals;
    Ptr<uint> uv0;
    Ptr<uint> uv1;
    Ptr<uint> uv2;
    Ptr<uint> uv3;
    Ptr<uint> colors;
    uint pointCount;
    //! The meshCorner bits of the mesh.
    uint cornerStreams;
    //! The rows of the transform of the first instance of the mesh.
    float4 objectToWorld[3];
    float4 worldToObject[3];
};

struct DisplacementGraphArguments {
    Ptr<DisplacementGraphRecord> record;
};

NR_NAMESPACE_END

#if !defined(__SLANG__)
static_assert(sizeof(NoorRay::Shaders::DisplacementRecord) == 16 * 8 + 32 + 32);
static_assert(sizeof(NoorRay::Shaders::DisplacementGraphRecord) == 10 * 8 + 8 + 96);
#endif
