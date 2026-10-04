#pragma once

// The hit stages every realtime material shares, generic over the material.
// SlangMaterialGenerator emits a material's closest-hit and any-hit entry
// points as calls into these, and DefaultMaterialHit.slang does the same for
// surfaces without a compiled material. Materials are compiled at runtime, so
// the build preprocesses this file, with everything it includes, into the one
// source the material compiler loads.

import MaterialInterface;

#include "HitPayload.slang"
#include "../Raytracer/SceneSurface.slang"

// Realtime launches push a RealtimeRoot, whose RealtimeArgs begin with the
// frame (RealtimeArgs.h asserts it). Reading only the frame keeps RTXDI's
// headers out of the material compile.
Frame* materialFrame()
{
    return (Frame*)(*(uint64_t*)root.arguments);
}

// The view camera, for the camera_position, camera_forward, camera_right and
// camera_up geometric properties. The camera looks down its local -Z.
float3 materialCameraPosition()
{
    const Camera camera = materialFrame()->camera;
    return float3(camera.cameraToWorld[3], camera.cameraToWorld[7], camera.cameraToWorld[11]);
}

float3 materialCameraForward()
{
    const Camera camera = materialFrame()->camera;
    return -normalize(float3(camera.cameraToWorld[2], camera.cameraToWorld[6], camera.cameraToWorld[10]));
}

float3 materialCameraRight()
{
    const Camera camera = materialFrame()->camera;
    return normalize(float3(camera.cameraToWorld[0], camera.cameraToWorld[4], camera.cameraToWorld[8]));
}

float3 materialCameraUp()
{
    const Camera camera = materialFrame()->camera;
    return normalize(float3(camera.cameraToWorld[1], camera.cameraToWorld[5], camera.cameraToWorld[9]));
}

// MaterialX's view and projection uniforms, in its viewer's conventions: a
// right-handed view space looking down -Z, GL clip space with an infinite far
// plane, and matrices applied to row vectors (see materialMatrix). The
// realtime renderer uses perspective projection.
float4x4 materialViewInverseMatrix()
{
    const Camera camera = materialFrame()->camera;
    return float4x4(float4(materialCameraRight(), 0.0),
        float4(materialCameraUp(), 0.0),
        float4(-materialCameraForward(), 0.0),
        float4(camera.cameraToWorld[3], camera.cameraToWorld[7], camera.cameraToWorld[11], 1.0));
}

float4x4 materialViewMatrix()
{
    const float4x4 cameraToWorld = materialViewInverseMatrix();
    const float3 position = cameraToWorld[3].xyz;
    // The inverse of a rotation is its transpose.
    const float3x3 rotation = transpose(float3x3(cameraToWorld[0].xyz, cameraToWorld[1].xyz,
        cameraToWorld[2].xyz));
    return float4x4(float4(rotation[0], 0.0), float4(rotation[1], 0.0),
        float4(rotation[2], 0.0), float4(-mul(position, rotation), 1.0));
}

// The film's half extents at unit distance, matching perspective ray
// generation.
float2 materialFilmScale(Camera camera)
{
    const float sensorWidth = max(camera.sensorWidthMm, 0.001);
    const float sensorHeight = max(camera.sensorHeightMm, 0.001);
    const float focalLength = max(camera.focalLengthMm, 0.001);
    return float2(sensorWidth, sensorHeight) / (2.0 * focalLength);
}

float4x4 materialProjectionMatrix()
{
    Frame* frame = materialFrame();
    const float2 scale = materialFilmScale(frame->camera);
    const float nearPlane = frame->nearPlane;
    return float4x4(float4(1.0 / scale.x, 0.0, 0.0, 0.0), float4(0.0, 1.0 / scale.y, 0.0, 0.0),
        float4(0.0, 0.0, -1.0, -1.0), float4(0.0, 0.0, -2.0 * nearPlane, 0.0));
}

float4x4 materialProjectionInverseMatrix()
{
    Frame* frame = materialFrame();
    const float2 scale = materialFilmScale(frame->camera);
    const float nearPlane = frame->nearPlane;
    return float4x4(float4(scale.x, 0.0, 0.0, 0.0), float4(0.0, scale.y, 0.0, 0.0),
        float4(0.0, 0.0, 0.0, -0.5 / nearPlane), float4(0.0, 0.0, -1.0, 0.5 / nearPlane));
}

// The camera_film_scale geometric property: the film's half extents at unit
// distance, the tangents of half the fields of view of a perspective camera.
float2 materialCameraFilmScale()
{
    return materialFilmScale(materialFrame()->camera);
}

// The view_size geometric property: the traced image's size in pixels.
float2 materialViewSize()
{
    Frame* frame = materialFrame();
    return float2(frame->width, frame->height);
}

// The custom_primitive_data_<index> geometric property: floats 4 index to
// 4 index + 3 of the hit instance's custom data, zero past its end.
float4 materialCustomPrimitiveData(uint index)
{
    Frame* frame = materialFrame();
    const Instance instance = frame->scene.instances[min(InstanceID(), frame->scene.instanceCount - 1u)];
    float4 result = float4(0.0);
    for (uint c = 0; c < 4u; ++c)
    {
        const uint word = index * 4u + c;
        if (word < instance.customDataCount)
            result[c] = instance.customData[word];
    }
    return result;
}

// Scene depth where nothing opaque lies behind.
static const float MaterialFarSceneDepth = 1.0e6;

// The scene_depth geometric property: the view depth of the first opaque
// surface behind this hit along the ray, as a rasterizer's depth buffer holds
// it under a translucent surface. Fractional and alpha-tested geometry runs
// materials to be resolved, so it is left out, as translucency is from a
// depth buffer.
float materialSceneDepth()
{
    RayDesc ray;
    ray.Origin = WorldRayOrigin();
    ray.Direction = WorldRayDirection();
    ray.TMin = asfloat(asuint(RayTCurrent()) + 1u);
    ray.TMax = 1.0e30;
    RayQuery<RAY_FLAG_CULL_NON_OPAQUE> query;
    query.TraceRayInline(RaytracingAccelerationStructure(materialFrame()->topLevelAS),
        RAY_FLAG_NONE, RaytracingMaskCamera, ray);
    query.Proceed();
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return MaterialFarSceneDepth;
    const float3 behind = ray.Origin + ray.Direction * query.CommittedRayT();
    return min(dot(behind - materialCameraPosition(), materialCameraForward()),
        MaterialFarSceneDepth);
}

// The surface a ray hit and its differentials, from the ray's differential.
struct MaterialHitSurface
{
    RegularSurface surface;
    SurfaceDifferentials differentials;
};

MaterialHitSurface hitSurface(Frame* frame, float2 barycentrics, VertexDifferential ray)
{
    const SurfaceTriangle triangle = loadSurfaceTriangle(frame->scene, InstanceID(),
        GeometryIndex(), PrimitiveIndex());
    MaterialHitSurface hit;
    hit.surface = interpolateSurface(triangle, barycentrics);
    hit.differentials = surfaceDifferentials(triangle, barycentrics, ray,
        WorldRayDirection(), RayTCurrent());
    return hit;
}

surfaceshader evaluateHit<M : IMaterial>(Frame* frame, MaterialHitSurface hit)
{
    const RegularSurface surface = hit.surface;
    const SurfaceDifferentials differentials = hit.differentials;
    materialSamplerHandle = frame->materialSampler;
    materialTextureFootprint = max(length(differentials.uvDx), length(differentials.uvDy))
        * exp2(frame->textureLodBias);
    MaterialGeometry geometry;
    geometry.position = surface.position;
    geometry.normal = surface.shadingNormal;
    geometry.tangent = surface.tangent;
    geometry.bitangent = normalize(cross(surface.shadingNormal, surface.tangent))
        * surface.bitangentSign;
    geometry.uv = surface.uv;
    geometry.uvs = surface.uvs;
    geometry.uvCount = surface.uvCount;
    geometry.uvVertices = surface.uvVertices;
    geometry.uvBarycentrics = surface.uvBarycentrics;
    geometry.color = surface.vertexColor;
    Instance* instance = &frame->scene.instances[min(InstanceID(), frame->scene.instanceCount - 1u)];
    Mesh* mesh = frame->scene.meshes[min(instance->meshIndex, frame->scene.meshCount - 1u)];
    geometry.boundsMin = float3(mesh->boundsMin[0], mesh->boundsMin[1], mesh->boundsMin[2]);
    geometry.boundsMax = float3(mesh->boundsMax[0], mesh->boundsMax[1], mesh->boundsMax[2]);
    geometry.positionDx = differentials.vertex.positionDx;
    geometry.positionDy = differentials.vertex.positionDy;
    geometry.normalDx = differentials.vertex.normalDx;
    geometry.normalDy = differentials.vertex.normalDy;
    geometry.tangentDx = differentials.tangentDx;
    geometry.tangentDy = differentials.tangentDy;
    geometry.bitangentDx = differentials.bitangentDx;
    geometry.bitangentDy = differentials.bitangentDy;
    geometry.uvDx = differentials.uvDx;
    geometry.uvDy = differentials.uvDy;
    geometry.uvBarycentricsDx = differentials.barycentricDx;
    geometry.uvBarycentricsDy = differentials.barycentricDy;
    geometry.colorDx = differentials.colorDx;
    geometry.colorDy = differentials.colorDy;
    return M.evaluate(geometry, frame->materials[surface.materialIndex]->shaderParameters);
}

void materialClosestHit<M : IMaterial>(inout RealtimeHitPayload payload, float2 barycentrics)
{
    Frame* frame = materialFrame();
    const MaterialHitSurface hit = hitSurface(frame, barycentrics,
        unpackVertexDifferential(payload.differential));
    const RegularSurface surface = hit.surface;
    const uint materialFlags = frame->materials[surface.materialIndex]->flags;
    Instance* instance = &frame->scene.instances[min(InstanceID(), frame->scene.instanceCount - 1u)];
    if ((materialFlags & MaterialFlagAnimated) != 0u
        || (instance->flags & InstanceFlagAnimated) != 0u)
        InterlockedExchange(frame->beautyAccumulation->animatedHit[frame->accumulationSlot], 1u);
    const surfaceshader shaded = evaluateHit<M>(frame, hit);
    // Unreal's one-sided materials emit nothing from their back.
    const bool emits = (materialFlags & MaterialFlagOneSided) == 0u
        || HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;
    packVertexDifferential(payload.differential, hit.differentials.vertex);
    RealtimeClosure closure;
    closure.diffuse = shaded.bsdf.diffuse;
    closure.roughness = shaded.bsdf.roughness();
    closure.specularF0 = shaded.bsdf.specularF0;
    closure.specularF90 = shaded.bsdf.specularF90;
    packRealtimeHit(payload, InstanceID(), RayTCurrent(), surface.position,
        surface.geometricNormal, shaded.bsdf.normal(surface.shadingNormal), closure,
        emits ? shaded.emission : float3(0.0));
    payload.opacity = saturate(shaded.opacity);
    payload.flags = (materialFlags & MaterialFlagSky) != 0u ? RealtimeHitSky : 0u;
}

// One-sided backfaces and empty coverage are discarded. Fractional surfaces are returned to the
// ray-generation stage, which composites all layers in distance order.
void materialAnyHit<M : IMaterial>(RealtimeHitPayload payload, float2 barycentrics)
{
    Frame* frame = materialFrame();
    const MaterialHitSurface hit = hitSurface(frame, barycentrics,
        unpackVertexDifferential(payload.differential));
    if (HitKind() == HIT_KIND_TRIANGLE_BACK_FACE
        && (frame->materials[hit.surface.materialIndex]->flags & MaterialFlagOneSided) != 0u)
    {
        IgnoreHit();
        return;
    }
    if (evaluateHit<M>(frame, hit).opacity <= 0.0)
        IgnoreHit();
}

// Shadow rays pass every fractional layer, multiplying its transmittance.
// Non-opaque geometry is built without duplicate any-hit invocations, so each
// layer counts once and the product is independent of traversal order. They
// carry no differential: a shadow ray is not seen through a pixel, so its
// layers are evaluated at a point, with the finest texture detail.
void materialShadowAnyHit<M : IMaterial>(inout RealtimeShadowPayload payload,
    float2 barycentrics)
{
    Frame* frame = materialFrame();
    if (!sectionCastsShadow(frame->scene, frame->materials, InstanceID(), GeometryIndex()))
    {
        IgnoreHit();
        return;
    }
    const MaterialHitSurface hit = hitSurface(frame, barycentrics, zeroVertexDifferential());
    if (HitKind() == HIT_KIND_TRIANGLE_BACK_FACE
        && (frame->materials[hit.surface.materialIndex]->flags & MaterialFlagOneSided) != 0u)
    {
        IgnoreHit();
        return;
    }
    payload.transmittance *= 1.0 - saturate(evaluateHit<M>(frame, hit).opacity);
    if (payload.transmittance < RealtimeMinTransmittance)
        AcceptHitAndEndSearch();
    else
        IgnoreHit();
}
