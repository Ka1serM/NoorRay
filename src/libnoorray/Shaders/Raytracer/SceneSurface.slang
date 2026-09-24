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

// Matches nr::vertex_color::packLinear: RGB is sRGB UNORM8, alpha is linear UNORM8.
float srgbToLinear(float value)
{
    return value <= 0.04045 ? value / 12.92
        : pow((value + 0.055) / 1.055, 2.4);
}

float4 unpackVertexColor(uint packed)
{
    const float4 encoded = float4(packed & 0xffu, (packed >> 8u) & 0xffu,
                                  (packed >> 16u) & 0xffu, packed >> 24u) / 255.0;
    return float4(srgbToLinear(encoded.r), srgbToLinear(encoded.g),
                  srgbToLinear(encoded.b), encoded.a);
}

// Each mesh section is its own BLAS geometry, whose PrimitiveIndex() counts
// from the section's first triangle; the section also names the material.
RegularSurface loadRegularSurface(Scene scene, uint instanceId, uint geometryIndex,
    uint primitiveIndex, float2 barycentric)
{
    instanceId = min(instanceId, scene.instanceCount - 1u);
    const Instance instance = scene.instances[instanceId];
    const uint meshIndex = min(instance.meshIndex, scene.meshCount - 1u);
    const Mesh mesh = *scene.meshes[meshIndex];
    const SectionRecord section = mesh.sections[min(geometryIndex, mesh.sectionCount - 1u)];

    const uint triangle = min(section.firstTriangle + primitiveIndex,
        max(mesh.indexCount / 3u, 1u) - 1u);
    const uint i0 = min(mesh.indices[triangle * 3u], mesh.vertexCount - 1u);
    const uint i1 = min(mesh.indices[triangle * 3u + 1u], mesh.vertexCount - 1u);
    const uint i2 = min(mesh.indices[triangle * 3u + 2u], mesh.vertexCount - 1u);

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
    surface.vertexColor = unpackVertexColor(vertex0.color) * w
        + unpackVertexColor(vertex1.color) * barycentric.x
        + unpackVertexColor(vertex2.color) * barycentric.y;
    surface.instanceIndex = instanceId;
    surface.primitiveIndex = triangle;
    surface.materialIndex = section.material;
    return surface;
}
