#pragma once

// Typed native-mesh access for the graphics API ray-tracing path. Meshes hold
// Unreal's vertex streams (Shared/Mesh.h); this code only performs element
// indexing, decoding and interpolation.

#include "RayDifferential.slang"

struct RegularSurface
{
    float3 position;
    float3 geometricNormal;
    float3 shadingNormal;
    float3 tangent;
    float bitangentSign;
    float2 uv;
    float2* uvs;
    uint uvCount;
    uint3 uvVertices;
    float2 uvBarycentrics;
    float4 vertexColor;
    uint instanceIndex;
    uint primitiveIndex;
    uint materialIndex;
};

// Unreal's FPackedNormal: four SNORM8 components, x in the low byte.
float4 unpackSnorm8x4(uint packed)
{
    const int4 values = int4(int(packed << 24u) >> 24, int(packed << 16u) >> 24,
                             int(packed << 8u) >> 24, int(packed) >> 24);
    return max(float4(values) / 127.0, -1.0);
}

// Unreal's FColor, linear UNORM8: B | G << 8 | R << 16 | A << 24.
float4 unpackVertexColor(uint packed)
{
    return float4((packed >> 16u) & 0xffu, (packed >> 8u) & 0xffu, packed & 0xffu,
                  packed >> 24u) / 255.0;
}

// Each mesh section is its own BLAS geometry, whose PrimitiveIndex() counts
// from the section's first triangle; the section names the instance's
// material slot. The hit triangle, its vertices in world space: fetched once
// for both the surface and its differentials.
struct SurfaceTriangle
{
    uint instanceIndex;
    uint triangle;
    uint materialIndex;
    uint3 indices;
    float3 positions[3];
    float3 normals[3];
    float3 tangents[3];
    float2 uvs[3];
    float4 colors[3];
    // cross(normal, tangent) times this is the world-space bitangent.
    float bitangentSign;
    float2* uvStream;
    uint uvCount;
};

// Whether a hit section blocks shadow rays: the section and its material
// both cast shadows.
bool sectionCastsShadow(Scene scene, Material** materials, uint instanceId, uint geometryIndex)
{
    Instance* instance = &scene.instances[min(instanceId, scene.instanceCount - 1u)];
    Mesh* mesh = scene.meshes[min(instance->meshIndex, scene.meshCount - 1u)];
    SectionRecord* section = &mesh->sections[min(geometryIndex, mesh->sectionCount - 1u)];
    return section->castsShadow != 0u
        && (materials[instance->materials[section->slot]]->flags & MaterialFlagNoShadows) == 0u;
}

// Must run in a hit stage: the transform is the hit TLAS record's.
SurfaceTriangle loadSurfaceTriangle(Scene scene, uint instanceId, uint geometryIndex,
    uint primitiveIndex)
{
    instanceId = min(instanceId, scene.instanceCount - 1u);
    const Instance instance = scene.instances[instanceId];
    const uint meshIndex = min(instance.meshIndex, scene.meshCount - 1u);
    const Mesh mesh = *scene.meshes[meshIndex];
    const SectionRecord section = mesh.sections[min(geometryIndex, mesh.sectionCount - 1u)];
    const float3x4 objectToWorld = ObjectToWorld3x4();
    const float3x3 linear = float3x3(objectToWorld[0].xyz, objectToWorld[1].xyz,
        objectToWorld[2].xyz);
    // Normals transform by the inverse transpose, so non-uniform scales keep
    // them perpendicular to the surface.
    const float3x4 worldToObject = WorldToObject3x4();
    const float3x3 normalMatrix = transpose(float3x3(worldToObject[0].xyz,
        worldToObject[1].xyz, worldToObject[2].xyz));
    uint* colors = instance.colors != nullptr ? instance.colors : mesh.colors;

    SurfaceTriangle result;
    result.instanceIndex = instanceId;
    result.triangle = min(section.firstTriangle + primitiveIndex,
        max(mesh.indexCount / 3u, 1u) - 1u);
    result.materialIndex = instance.materials[section.slot];
    result.uvStream = mesh.uvs;
    result.uvCount = mesh.uvCount;
    [unroll]
    for (uint i = 0u; i < 3u; ++i)
    {
        const uint vertex = min(mesh.indices[result.triangle * 3u + i], mesh.vertexCount - 1u);
        const uint2 tangentFrame = mesh.tangents[vertex];
        const float4 tangentZ = unpackSnorm8x4(tangentFrame.y);
        result.indices[i] = vertex;
        result.positions[i] = mul(objectToWorld, float4(mesh.positions[vertex], 1.0));
        result.normals[i] = mul(normalMatrix, tangentZ.xyz);
        result.tangents[i] = mul(linear, unpackSnorm8x4(tangentFrame.x).xyz);
        result.uvs[i] = mesh.uvs[vertex * mesh.uvCount];
        result.colors[i] = colors != nullptr ? unpackVertexColor(colors[vertex]) : float4(1.0);
        if (i == 0u)
            // A mirroring transform flips the cross product of the
            // transformed normal and tangent, as Unreal's determinant sign does.
            result.bitangentSign = (tangentZ.w < 0.0 ? -1.0 : 1.0)
                * (determinant(linear) < 0.0 ? -1.0 : 1.0);
    }
    return result;
}

float2 interpolate(float2 a, float2 b, float2 c, float2 barycentric)
{
    return a * (1.0 - barycentric.x - barycentric.y) + b * barycentric.x + c * barycentric.y;
}

float3 interpolate(float3 a, float3 b, float3 c, float2 barycentric)
{
    return a * (1.0 - barycentric.x - barycentric.y) + b * barycentric.x + c * barycentric.y;
}

float4 interpolate(float4 a, float4 b, float4 c, float2 barycentric)
{
    return a * (1.0 - barycentric.x - barycentric.y) + b * barycentric.x + c * barycentric.y;
}

RegularSurface interpolateSurface(SurfaceTriangle triangle, float2 barycentric)
{
    const float3 p0 = triangle.positions[0];
    const float3 p1 = triangle.positions[1];
    const float3 p2 = triangle.positions[2];
    RegularSurface surface;
    surface.position = interpolate(p0, p1, p2, barycentric);
    surface.geometricNormal = normalize(cross(p1 - p0, p2 - p0));
    surface.shadingNormal = normalize(interpolate(triangle.normals[0], triangle.normals[1],
        triangle.normals[2], barycentric));
    surface.tangent = normalize(interpolate(triangle.tangents[0], triangle.tangents[1],
        triangle.tangents[2], barycentric));
    surface.bitangentSign = triangle.bitangentSign;
    surface.uv = interpolate(triangle.uvs[0], triangle.uvs[1], triangle.uvs[2], barycentric);
    surface.uvs = triangle.uvStream;
    surface.uvCount = triangle.uvCount;
    surface.uvVertices = triangle.indices;
    surface.uvBarycentrics = barycentric;
    surface.vertexColor = interpolate(triangle.colors[0], triangle.colors[1],
        triangle.colors[2], barycentric);
    surface.instanceIndex = triangle.instanceIndex;
    surface.primitiveIndex = triangle.triangle;
    surface.materialIndex = triangle.materialIndex;
    return surface;
}

RegularSurface loadRegularSurface(Scene scene, uint instanceId, uint geometryIndex,
    uint primitiveIndex, float2 barycentric)
{
    return interpolateSurface(loadSurfaceTriangle(scene, instanceId, geometryIndex,
        primitiveIndex), barycentric);
}

// Screen-space derivatives of what is interpolated over a hit, per render
// pixel along the image's x and y.
struct SurfaceDifferentials
{
    // The hit's position and shading normal, and the arriving ray's direction.
    VertexDifferential vertex;
    float2 barycentricDx;
    float2 barycentricDy;
    float3 tangentDx;
    float3 tangentDy;
    float3 bitangentDx;
    float3 bitangentDy;
    float2 uvDx;
    float2 uvDy;
    float4 colorDx;
    float4 colorDy;
};

// The barycentric change that moves a point of the triangle by `positionDelta`,
// which lies in its plane.
float2 barycentricDelta(float3 edge1, float3 edge2, float3 positionDelta)
{
    const float3 areaNormal = cross(edge1, edge2);
    const float areaSquared = dot(areaNormal, areaNormal);
    if (areaSquared <= 1.0e-20)
        return float2(0.0);
    return float2(dot(cross(positionDelta, edge2), areaNormal),
        dot(cross(edge1, positionDelta), areaNormal)) / areaSquared;
}

float2 interpolatedDelta(float2 a, float2 b, float2 c, float2 barycentricDelta)
{
    return (b - a) * barycentricDelta.x + (c - a) * barycentricDelta.y;
}

float3 interpolatedDelta(float3 a, float3 b, float3 c, float2 barycentricDelta)
{
    return (b - a) * barycentricDelta.x + (c - a) * barycentricDelta.y;
}

float4 interpolatedDelta(float4 a, float4 b, float4 c, float2 barycentricDelta)
{
    return (b - a) * barycentricDelta.x + (c - a) * barycentricDelta.y;
}

// The change of normalize(v) when v changes by `delta`.
float3 normalizedDelta(float3 v, float3 delta)
{
    const float size = max(length(v), 1.0e-12);
    const float3 unit = v / size;
    return (delta - unit * dot(unit, delta)) / size;
}

// The differentials of the ray `ray`, travelling along `direction`, where it
// hits `triangle` at distance t.
SurfaceDifferentials surfaceDifferentials(SurfaceTriangle triangle, float2 barycentric,
    VertexDifferential ray, float3 direction, float t)
{
    const float3 edge1 = triangle.positions[1] - triangle.positions[0];
    const float3 edge2 = triangle.positions[2] - triangle.positions[0];
    SurfaceDifferentials result;
    result.vertex = arriveAt(ray, direction, t, normalize(cross(edge1, edge2)));
    result.barycentricDx = barycentricDelta(edge1, edge2, result.vertex.positionDx);
    result.barycentricDy = barycentricDelta(edge1, edge2, result.vertex.positionDy);

    const float3 normal = interpolate(triangle.normals[0], triangle.normals[1],
        triangle.normals[2], barycentric);
    const float3 tangent = interpolate(triangle.tangents[0], triangle.tangents[1],
        triangle.tangents[2], barycentric);
    const float3 normalDx = interpolatedDelta(triangle.normals[0], triangle.normals[1],
        triangle.normals[2], result.barycentricDx);
    const float3 normalDy = interpolatedDelta(triangle.normals[0], triangle.normals[1],
        triangle.normals[2], result.barycentricDy);
    const float3 tangentDx = interpolatedDelta(triangle.tangents[0], triangle.tangents[1],
        triangle.tangents[2], result.barycentricDx);
    const float3 tangentDy = interpolatedDelta(triangle.tangents[0], triangle.tangents[1],
        triangle.tangents[2], result.barycentricDy);
    result.vertex.normalDx = normalizedDelta(normal, normalDx);
    result.vertex.normalDy = normalizedDelta(normal, normalDy);
    result.tangentDx = normalizedDelta(tangent, tangentDx);
    result.tangentDy = normalizedDelta(tangent, tangentDy);

    // The bitangent is normalize(cross(normal, tangent)) times the tangent sign.
    const float3 unitNormal = normalize(normal);
    const float3 unitTangent = normalize(tangent);
    const float3 bitangent = cross(unitNormal, unitTangent);
    const float sign = triangle.bitangentSign;
    result.bitangentDx = sign * normalizedDelta(bitangent, cross(result.vertex.normalDx, unitTangent)
        + cross(unitNormal, result.tangentDx));
    result.bitangentDy = sign * normalizedDelta(bitangent, cross(result.vertex.normalDy, unitTangent)
        + cross(unitNormal, result.tangentDy));

    result.uvDx = interpolatedDelta(triangle.uvs[0], triangle.uvs[1], triangle.uvs[2],
        result.barycentricDx);
    result.uvDy = interpolatedDelta(triangle.uvs[0], triangle.uvs[1], triangle.uvs[2],
        result.barycentricDy);
    result.colorDx = interpolatedDelta(triangle.colors[0], triangle.colors[1], triangle.colors[2],
        result.barycentricDx);
    result.colorDy = interpolatedDelta(triangle.colors[0], triangle.colors[1], triangle.colors[2],
        result.barycentricDy);
    return result;
}
