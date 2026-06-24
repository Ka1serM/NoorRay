#ifndef SURFACE_DATA_SLANG
#define SURFACE_DATA_SLANG

#include "../Shared.h"
#include "../Bindings.slang"
#include "../Materials/PbrMaterial.slang"
#include "Intersection.slang"

struct SurfaceData
{
    float3 worldPosition;
    float3 worldShadowPosition;
    float3 shadingNormal;
    float3 geometricNormal;
    float3 albedo;
    float3 emission;
    float metallic;
    float specular;
    float roughness;
    float opacity;
    float transmission;
    float ior;
    float3 transmissionColor;
};

SurfaceData loadComputeSurfaceData(uint instanceIndex, uint primitiveIndex, float3 barycentrics)
{
    SurfaceData surface;

    const Instance inst = instances[instanceIndex];
    const MeshAddresses mesh = meshes[inst.meshId];
    const Face face = reinterpret<FaceBuffer*>(mesh.faceAddress)->data[primitiveIndex];
    const Material material = reinterpret<MaterialBuffer*>(mesh.materialAddress)->data[face.materialIndex];

    const Vertex v0 = reinterpret<VertexBuffer*>(mesh.vertexAddress)->data[reinterpret<IndexBuffer*>(mesh.indexAddress)->data[3 * primitiveIndex + 0]];
    const Vertex v1 = reinterpret<VertexBuffer*>(mesh.vertexAddress)->data[reinterpret<IndexBuffer*>(mesh.indexAddress)->data[3 * primitiveIndex + 1]];
    const Vertex v2 = reinterpret<VertexBuffer*>(mesh.vertexAddress)->data[reinterpret<IndexBuffer*>(mesh.indexAddress)->data[3 * primitiveIndex + 2]];

    float3 localPos = interpolateBarycentric(barycentrics, v0.position, v1.position, v2.position);
    float3 localShadowPos = computeShadowTerminatorPointLocal(
        localPos,
        barycentrics,
        v0.position,
        v1.position,
        v2.position,
        v0.normal,
        v1.normal,
        v2.normal);
    float3 geometricNormalLocal = normalize(cross(v1.position - v0.position, v2.position - v0.position));
    float3 shadingNormalLocal = normalize(interpolateBarycentric(barycentrics, v0.normal, v1.normal, v2.normal));
    float3 localTan = normalize(interpolateBarycentric(barycentrics, v0.tangent, v1.tangent, v2.tangent));
    float tangentSign = dot(float3(v0.tangentSign, v1.tangentSign, v2.tangentSign), barycentrics);
    float2 uv = interpolateBarycentric(barycentrics, v0.uv, v1.uv, v2.uv);

    samplePbrMaterial(
        material,
        uv,
        surface.albedo,
        surface.emission,
        surface.metallic,
        surface.specular,
        surface.roughness,
        surface.opacity,
        surface.transmission);
    surface.ior = material.ior;
    surface.transmissionColor = material.transmissionColor;

    surface.worldPosition = mul(inst.transform, float4(localPos, 1.0)).xyz;
    surface.worldShadowPosition = mul(inst.transform, float4(localShadowPos, 1.0)).xyz;

    float3x3 normalMatrix = (float3x3)inst.normalTransform;
    surface.geometricNormal = normalize(mul(normalMatrix, geometricNormalLocal));
    surface.shadingNormal = sampleMaterialShadingNormal(
        material.normalIndex,
        uv,
        normalize(mul(normalMatrix, shadingNormalLocal)),
        normalize(mul(normalMatrix, localTan)),
        tangentSign);

    return surface;
}

#ifdef USE_RTX
SurfaceData loadRayTracingSurfaceData(uint tlasInstanceIndex, uint primitiveIndex, float2 hitBary)
{
    SurfaceData surface;

    const float3 bary = float3(1.0 - hitBary.x - hitBary.y, hitBary.x, hitBary.y);
    const Instance inst = instances[tlasInstanceIndex];
    const MeshAddresses mesh = meshes[inst.meshId];

    const Face face = reinterpret<FaceBuffer*>(mesh.faceAddress)->data[primitiveIndex];
    const Material material = reinterpret<MaterialBuffer*>(mesh.materialAddress)->data[face.materialIndex];

    const Vertex v0 = reinterpret<VertexBuffer*>(mesh.vertexAddress)->data[reinterpret<IndexBuffer*>(mesh.indexAddress)->data[3 * primitiveIndex + 0]];
    const Vertex v1 = reinterpret<VertexBuffer*>(mesh.vertexAddress)->data[reinterpret<IndexBuffer*>(mesh.indexAddress)->data[3 * primitiveIndex + 1]];
    const Vertex v2 = reinterpret<VertexBuffer*>(mesh.vertexAddress)->data[reinterpret<IndexBuffer*>(mesh.indexAddress)->data[3 * primitiveIndex + 2]];

    float3 localPosition = interpolateBarycentric(bary, v0.position, v1.position, v2.position);
    float3 localShadowPosition = computeShadowTerminatorPointLocal(
        localPosition,
        bary,
        v0.position,
        v1.position,
        v2.position,
        v0.normal,
        v1.normal,
        v2.normal);
    float3 geometricNormalLocal = normalize(cross(v1.position - v0.position, v2.position - v0.position));
    float3 shadingNormalLocal = normalize(interpolateBarycentric(bary, v0.normal, v1.normal, v2.normal));
    float3 localTangent = normalize(interpolateBarycentric(bary, v0.tangent, v1.tangent, v2.tangent));
    float tangentSign = dot(float3(v0.tangentSign, v1.tangentSign, v2.tangentSign), bary);
    float2 uv = interpolateBarycentric(bary, v0.uv, v1.uv, v2.uv);

    samplePbrMaterial(
        material,
        uv,
        surface.albedo,
        surface.emission,
        surface.metallic,
        surface.specular,
        surface.roughness,
        surface.opacity,
        surface.transmission);
    surface.ior = material.ior;
    surface.transmissionColor = material.transmissionColor;

    float4x3 objectToWorld = ObjectToWorld4x3();
    surface.worldPosition = mul(float4(localPosition, 1.0), objectToWorld);
    surface.worldShadowPosition = mul(float4(localShadowPosition, 1.0), objectToWorld);

    float3x3 normalMatrix = (float3x3)inst.normalTransform;
    surface.geometricNormal = normalize(mul(normalMatrix, geometricNormalLocal));
    surface.shadingNormal = sampleMaterialShadingNormal(
        material.normalIndex,
        uv,
        normalize(mul(normalMatrix, shadingNormalLocal)),
        normalize(mul(normalMatrix, localTangent)),
        tangentSign);

    return surface;
}
#endif

#endif // SURFACE_DATA_SLANG
