#pragma once

// Typed native-mesh access for the graphics API ray-tracing path. The host uploads
// separate arrays for records, vertices, indices, faces, and material IDs;
// this code only performs element indexing and interpolation.

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

float3 transformPoint(Instance instance, float3 p)
{
    return float3(
        dot(float4(p, 1.0), float4(instance.objectToWorld[0],
            instance.objectToWorld[1], instance.objectToWorld[2],
            instance.objectToWorld[3])),
        dot(float4(p, 1.0), float4(instance.objectToWorld[4],
            instance.objectToWorld[5], instance.objectToWorld[6],
            instance.objectToWorld[7])),
        dot(float4(p, 1.0), float4(instance.objectToWorld[8],
            instance.objectToWorld[9], instance.objectToWorld[10],
            instance.objectToWorld[11])));
}

float3 transformVector(Instance instance, float3 p)
{
    return float3(
        dot(p, float3(instance.objectToWorld[0], instance.objectToWorld[1],
            instance.objectToWorld[2])),
        dot(p, float3(instance.objectToWorld[4], instance.objectToWorld[5],
            instance.objectToWorld[6])),
        dot(p, float3(instance.objectToWorld[8], instance.objectToWorld[9],
            instance.objectToWorld[10])));
}

RegularSurface loadRegularSurface(Scene scene,
    uint instanceId, uint primitiveId, float2 barycentric)
{
    instanceId = min(instanceId, scene.instanceCount - 1u);
    const Instance instance = scene.instances[instanceId];
    const uint meshIndex = min(instance.meshIndex, scene.meshCount - 1u);
    const Mesh mesh = *scene.meshes[meshIndex];

    primitiveId = min(primitiveId, max(mesh.indexCount / 3u, 1u) - 1u);
    uint i0 = primitiveId * 3u;
    uint i1 = i0 + 1u;
    uint i2 = i0 + 2u;
    if (mesh.indexCount != 0u)
    {
        i0 = mesh.indices[i0];
        i1 = mesh.indices[i1];
        i2 = mesh.indices[i2];
    }
    i0 = min(i0, mesh.vertexCount - 1u);
    i1 = min(i1, mesh.vertexCount - 1u);
    i2 = min(i2, mesh.vertexCount - 1u);

    const Vertex vertex0 = mesh.vertices[i0];
    const Vertex vertex1 = mesh.vertices[i1];
    const Vertex vertex2 = mesh.vertices[i2];
    const float w = 1.0 - barycentric.x - barycentric.y;
    const float3 p0 = transformPoint(instance, vertex0.position);
    const float3 p1 = transformPoint(instance, vertex1.position);
    const float3 p2 = transformPoint(instance, vertex2.position);
    const float3 n0 = transformVector(instance, vertex0.normal);
    const float3 n1 = transformVector(instance, vertex1.normal);
    const float3 n2 = transformVector(instance, vertex2.normal);
    const float3 t0 = transformVector(instance, vertex0.tangent);
    const float3 t1 = transformVector(instance, vertex1.tangent);
    const float3 t2 = transformVector(instance, vertex2.tangent);

    RegularSurface surface;
    surface.position = p0 * w + p1 * barycentric.x + p2 * barycentric.y;
    surface.geometricNormal = normalize(cross(p1 - p0, p2 - p0));
    surface.shadingNormal = normalize(n0 * w + n1 * barycentric.x + n2 * barycentric.y);
    surface.tangent = normalize(t0 * w + t1 * barycentric.x + t2 * barycentric.y);
    surface.tangentSign = vertex0.tangentSign;
    surface.uv = vertex0.uv * w + vertex1.uv * barycentric.x
        + vertex2.uv * barycentric.y;
    surface.vertexColor = vertex0.color * w + vertex1.color * barycentric.x
        + vertex2.color * barycentric.y;
    surface.instanceIndex = instanceId;
    surface.primitiveIndex = primitiveId;

    uint localMaterial = 0u;
    if (mesh.faceCount != 0u)
    {
        const uint faceIndex = min(primitiveId, mesh.faceCount - 1u);
        const int authoredMaterial = mesh.faces[faceIndex].materialIndex;
        localMaterial = authoredMaterial >= 0 ? uint(authoredMaterial) : 0u;
        localMaterial = min(localMaterial, max(mesh.materialCount, 1u) - 1u);
    }
    surface.materialIndex = mesh.materialCount != 0u
        ? mesh.materialIds[localMaterial] : 0u;
    return surface;
}
