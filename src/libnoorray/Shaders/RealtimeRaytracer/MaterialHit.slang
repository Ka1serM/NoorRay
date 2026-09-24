#pragma once

// The hit stages every realtime material shares, generic over the material.
// SlangMaterialGenerator emits a material's closest-hit and any-hit entry
// points as calls into these, and DefaultMaterialHit.slang does the same for
// surfaces without a compiled material. Materials are compiled at runtime, so
// the build preprocesses this file, with everything it includes, into the one
// source the material compiler loads.

import MaterialInterface;

#include "../Raytracer/RaytracingAbi.slang"
#include "../Raytracer/SceneSurface.slang"

// Realtime launches push a RealtimeRoot, whose RealtimeArgs begin with the
// frame (RealtimeArgs.h asserts it). Reading only the frame keeps RTXDI's
// headers out of the material compile.
Frame* materialFrame()
{
    return (Frame*)(*(uint64_t*)root.arguments);
}

RegularSurface hitSurface(Frame* frame, float2 barycentrics)
{
    return loadRegularSurface(frame->scene, InstanceID(), GeometryIndex(),
        PrimitiveIndex(), barycentrics);
}

surfaceshader evaluateHit<M : IMaterial>(Frame* frame, RegularSurface surface)
{
    MaterialGeometry geometry;
    geometry.position = surface.position;
    geometry.normal = surface.shadingNormal;
    geometry.tangent = surface.tangent;
    geometry.bitangent = normalize(cross(surface.shadingNormal, surface.tangent))
        * surface.tangentSign;
    geometry.uv = surface.uv;
    geometry.color = surface.vertexColor;
    return M.evaluate(geometry, frame->materials[surface.materialIndex]->shaderParameters);
}

void materialClosestHit<M : IMaterial>(inout RealtimeHitPayload payload, float2 barycentrics)
{
    Frame* frame = materialFrame();
    const RegularSurface surface = hitSurface(frame, barycentrics);
    const surfaceshader shaded = evaluateHit<M>(frame, surface);
    payload.hit = 1u;
    payload.instance = InstanceID();
    payload.t = RayTCurrent();
    payload.position = surface.position;
    payload.geometricNormal = surface.geometricNormal;
    payload.shadingNormal = surface.shadingNormal;
    payload.diffuse = shaded.bsdf.diffuse;
    payload.specularF0 = shaded.bsdf.specularF0;
    payload.specularF90 = shaded.bsdf.specularF90;
    payload.roughnessSum = shaded.bsdf.roughnessSum;
    payload.roughnessWeight = shaded.bsdf.roughnessWeight;
    payload.normalSum = shaded.bsdf.normalSum;
    payload.emission = shaded.emission;
}

// Opacity is an alpha test, as Unreal's masked blend mode applies it: a
// candidate below the cutoff is ignored, anything else blocks the ray. Only
// the opacity of the material's result is used, so the compiler drops
// everything else the material computes.
static const float OpacityCutoff = 1.0 / 3.0; // Unreal's default OpacityMaskClipValue.

void materialAnyHit<M : IMaterial>(float2 barycentrics)
{
    Frame* frame = materialFrame();
    if (evaluateHit<M>(frame, hitSurface(frame, barycentrics)).opacity < OpacityCutoff)
        IgnoreHit();
}
