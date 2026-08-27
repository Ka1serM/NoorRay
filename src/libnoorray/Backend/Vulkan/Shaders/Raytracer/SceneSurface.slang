// Packed-scene surface decoding for the Vulkan ray-tracing path.
//
// The host writes meshes into one immutable byte buffer (see VulkanScene);
// this file turns a hit's instance/primitive/barycentrics into the interpolated
// surface the closest-hit shader shades. It declares no descriptors, so the
// mesh, gaussian and differentiable paths can all share it.

// Slang-side regular mesh path.
//
// This is the Vulkan counterpart of the ordinary mesh closest-hit path.  The
// transport policy is deliberately expressed as interfaces: scene upload,
// SVM material evaluation, light-tree sampling, and ray visibility can each
// provide a concrete implementation without changing the dispatch ABI in
// Raytracer.slang.  Keeping this file free of descriptor declarations also
// makes it usable by the gaussian and differentiable paths.

struct RegularSurface
{
    float3 position;
    float3 geometricNormal;
    float3 shadingNormal;
    float3 tangent;
    float tangentSign;
    float2 uv;
    float4 vertexColor;
    uint instanceIndex;
    uint primitiveIndex;
    uint materialIndex;
};

struct PackedSceneHeader
{
    uint meshCount;
    uint instanceCount;
    uint meshOffset;
    uint instanceOffset;
    uint vertexDataOffset;
    uint indexDataOffset;
    uint faceDataOffset;
    uint totalBytes;
};

struct PackedMeshRecord
{
    uint vertexOffset;
    uint indexOffset;
    uint faceOffset;
    uint vertexCount;
    uint indexCount;
    uint faceCount;
    uint materialOffset;
    uint materialCount;
};

uint packedLoad(ByteAddressBuffer scene, uint offset)
{
    return scene.Load(offset);
}

float packedFloat(ByteAddressBuffer scene, uint offset)
{
    return asfloat(packedLoad(scene, offset));
}

float3 packedVertexPosition(ByteAddressBuffer scene, uint offset)
{
    return float3(packedFloat(scene, offset), packedFloat(scene, offset + 4),
        packedFloat(scene, offset + 8));
}

float3 packedVertexNormal(ByteAddressBuffer scene, uint offset)
{
    return float3(packedFloat(scene, offset + 12), packedFloat(scene, offset + 16),
        packedFloat(scene, offset + 20));
}

float3 packedVertexTangent(ByteAddressBuffer scene, uint offset)
{
    return float3(packedFloat(scene, offset + 24), packedFloat(scene, offset + 28),
        packedFloat(scene, offset + 32));
}

float2 packedVertexUv(ByteAddressBuffer scene, uint offset)
{
    return float2(packedFloat(scene, offset + 40), packedFloat(scene, offset + 44));
}

float4 packedVertexColor(ByteAddressBuffer scene, uint offset)
{
    const uint packed = packedLoad(scene, offset + 48);
    return float4(float((packed >> 0) & 0xFFu), float((packed >> 8) & 0xFFu),
        float((packed >> 16) & 0xFFu), float((packed >> 24) & 0xFFu)) / 255.0;
}

float3 packedTransformPoint(ByteAddressBuffer scene, uint offset, float3 p)
{
    return float3(
        dot(float4(p, 1.0), float4(packedFloat(scene, offset + 0),
            packedFloat(scene, offset + 4), packedFloat(scene, offset + 8),
            packedFloat(scene, offset + 12))),
        dot(float4(p, 1.0), float4(packedFloat(scene, offset + 16),
            packedFloat(scene, offset + 20), packedFloat(scene, offset + 24),
            packedFloat(scene, offset + 28))),
        dot(float4(p, 1.0), float4(packedFloat(scene, offset + 32),
            packedFloat(scene, offset + 36), packedFloat(scene, offset + 40),
            packedFloat(scene, offset + 44))));
}

float3 packedTransformVector(ByteAddressBuffer scene, uint offset, float3 p)
{
    return float3(
        dot(p, float3(packedFloat(scene, offset + 0), packedFloat(scene, offset + 4),
            packedFloat(scene, offset + 8))),
        dot(p, float3(packedFloat(scene, offset + 16), packedFloat(scene, offset + 20),
            packedFloat(scene, offset + 24))),
        dot(p, float3(packedFloat(scene, offset + 32), packedFloat(scene, offset + 36),
            packedFloat(scene, offset + 40))));
}

RegularSurface loadRegularSurface(ByteAddressBuffer scene,
    uint instanceId, uint primitiveId, float2 barycentric)
{
    PackedSceneHeader header;
    header.meshCount = packedLoad(scene, 0);
    header.instanceCount = packedLoad(scene, 4);
    header.meshOffset = packedLoad(scene, 8);
    header.instanceOffset = packedLoad(scene, 12);
    header.vertexDataOffset = packedLoad(scene, 16);
    header.indexDataOffset = packedLoad(scene, 20);
    header.faceDataOffset = packedLoad(scene, 24);
    header.totalBytes = packedLoad(scene, 28);
    instanceId = min(instanceId, header.instanceCount - 1u);
    const uint instanceOffset = header.instanceOffset + instanceId * 64u;
    uint meshIndex = packedLoad(scene, instanceOffset + 48);
    meshIndex = min(meshIndex, header.meshCount - 1u);
    const uint meshOffset = header.meshOffset + meshIndex * 32u;
    const uint vertexOffset = packedLoad(scene, meshOffset + 0);
    const uint indexOffset = packedLoad(scene, meshOffset + 4);
    const uint vertexCount = packedLoad(scene, meshOffset + 12);
    const uint indexCount = packedLoad(scene, meshOffset + 16);
    const uint faceCount = packedLoad(scene, meshOffset + 20);
    const uint materialOffset = packedLoad(scene, meshOffset + 24);
    const uint materialCount = packedLoad(scene, meshOffset + 28);
    primitiveId = min(primitiveId, max(indexCount / 3u, 1u) - 1u);
    uint i0 = primitiveId * 3u;
    uint i1 = i0 + 1u;
    uint i2 = i0 + 2u;
    if (indexCount != 0)
    {
        i0 = packedLoad(scene, indexOffset + i0 * 4u);
        i1 = packedLoad(scene, indexOffset + i1 * 4u);
        i2 = packedLoad(scene, indexOffset + i2 * 4u);
    }
    i0 = min(i0, vertexCount - 1u);
    i1 = min(i1, vertexCount - 1u);
    i2 = min(i2, vertexCount - 1u);
    const uint vertexStride = 52u;
    const uint v0Offset = vertexOffset + i0 * vertexStride;
    const uint v1Offset = vertexOffset + i1 * vertexStride;
    const uint v2Offset = vertexOffset + i2 * vertexStride;
    const float w = 1.0 - barycentric.x - barycentric.y;
    const float3 p0 = packedTransformPoint(scene, instanceOffset,
        packedVertexPosition(scene, v0Offset));
    const float3 p1 = packedTransformPoint(scene, instanceOffset,
        packedVertexPosition(scene, v1Offset));
    const float3 p2 = packedTransformPoint(scene, instanceOffset,
        packedVertexPosition(scene, v2Offset));
    const float3 n0 = packedTransformVector(scene, instanceOffset,
        packedVertexNormal(scene, v0Offset));
    const float3 n1 = packedTransformVector(scene, instanceOffset,
        packedVertexNormal(scene, v1Offset));
    const float3 n2 = packedTransformVector(scene, instanceOffset,
        packedVertexNormal(scene, v2Offset));
    const float3 t0 = packedTransformVector(scene, instanceOffset,
        packedVertexTangent(scene, v0Offset));
    const float3 t1 = packedTransformVector(scene, instanceOffset,
        packedVertexTangent(scene, v1Offset));
    const float3 t2 = packedTransformVector(scene, instanceOffset,
        packedVertexTangent(scene, v2Offset));
    RegularSurface surface;
    surface.position = p0 * w + p1 * barycentric.x + p2 * barycentric.y;
    surface.geometricNormal = normalize(cross(p1 - p0, p2 - p0));
    surface.shadingNormal = normalize(n0 * w + n1 * barycentric.x + n2 * barycentric.y);
    surface.tangent = normalize(t0 * w + t1 * barycentric.x + t2 * barycentric.y);
    surface.tangentSign = packedFloat(scene, v0Offset + 36);
    surface.uv = packedVertexUv(scene, v0Offset) * w
        + packedVertexUv(scene, v1Offset) * barycentric.x
        + packedVertexUv(scene, v2Offset) * barycentric.y;
    surface.vertexColor = packedVertexColor(scene, v0Offset) * w
        + packedVertexColor(scene, v1Offset) * barycentric.x
        + packedVertexColor(scene, v2Offset) * barycentric.y;
    surface.instanceIndex = instanceId;
    surface.primitiveIndex = primitiveId;
    uint localMaterial = 0u;
    if (faceCount != 0u)
    {
        const uint faceIndex = min(primitiveId, faceCount - 1u);
        localMaterial = min(packedLoad(scene, packedLoad(scene, meshOffset + 8u)
            + faceIndex * 4u), max(materialCount, 1u) - 1u);
    }
    surface.materialIndex = materialCount != 0u
        ? packedLoad(scene, materialOffset + localMaterial * 4u) : 0u;
    return surface;
}
