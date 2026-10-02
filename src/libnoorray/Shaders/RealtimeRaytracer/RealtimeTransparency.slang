#pragma once

#include "RealtimeHitShading.slang"

// Fractional surfaces are layers a ray passes straight through. Primary
// visibility keeps the nearest one as a surface of its own, which ReSTIR and
// NRD light like the G-buffer's, and walks the others to the first opaque
// surface for the usual G-buffer and ReSTIR path.
static const uint MaxTranslucentLayers = 8u;

// Advancing the original ray by one representable distance avoids both
// re-hitting the current layer and stepping across a nearby second layer.
float realtimeNextLayer(float distance)
{
    return asfloat(asuint(distance) + 1u);
}

// A layer is taken as opaque past MaxTranslucentLayers or once it would pass
// less than RealtimeMinTransmittance.
bool realtimeLayerEndsRay(RealtimeHitPayload payload, uint layer, float transmittance)
{
    const float opacity = saturate(payload.opacity);
    return payload.instance == RealtimeMissInstance || opacity >= 1.0
        || layer == MaxTranslucentLayers
        || transmittance * (1.0 - opacity) < RealtimeMinTransmittance;
}

struct TranslucentLayers
{
    // The first opaque hit or the miss.
    RealtimeHitPayload surface;
    // Fraction of the surface's light that reaches the ray's origin.
    float transmittance;
};

// An Unreal "Is Sky" surface found after a bounce ends the path, as in Unreal's
// path tracer, opaque and without a closure. Its light is the sky light's: a
// specified cubemap carries it, so the surface is black; a captured scene is
// what such surfaces emit, so the surface shows that light as the sky light
// would cast it, from the direction it is seen in.
void absorbSky(inout RealtimeHitPayload payload, float3 direction, float pathRoughness)
{
    if ((payload.flags & RealtimeHitSky) == 0u || payload.instance == RealtimeMissInstance)
        return;
    const RealtimeHit hit = unpackRealtimeHit(payload);
    float3 emission = float3(0.0);
    Frame frame = realtimeArgs()->frame;
    if (frame.environment != nullptr && frame.environment[0].capturedScene.a > 0.0)
    {
        const Environment environment = frame.environment[0];
        emission = max(hit.emission, 0.0);
        if (environment.lowerHemisphere.a > 0.0 && environmentHeight(environment, direction) < 0.0)
            emission = lerp(emission, environment.lowerHemisphere.rgb, environment.lowerHemisphere.a);
        emission *= environment.capturedScene.rgb * environment.lightingExposureScale
            * lerp(1.0, environment.indirectLightingIntensity, saturate(4.0 * pathRoughness));
    }
    packRealtimeHit(payload, hit.instance, hit.t, hit.position, hit.geometricNormal, hit.normal,
        (RealtimeClosure)0, emission);
    payload.opacity = 1.0;
}

// The layers of a ray after a bounce from a surface reached with
// `pathRoughness`. Every layer is found by the same ray, so each sees its
// differential.
TranslucentLayers traceTranslucentLayers(RayDesc ray, VertexDifferential differential,
    float pathRoughness)
{
    TranslucentLayers layers;
    layers.transmittance = 1.0;
    [loop]
    for (uint layer = 1u; ; ++layer)
    {
        layers.surface = traceRealtimeHitPayload(ray, differential, RaytracingMaskIndirect,
            RAY_FLAG_NONE);
        absorbSky(layers.surface, ray.Direction, pathRoughness);
        if (realtimeLayerEndsRay(layers.surface, layer, layers.transmittance))
            return layers;
        layers.transmittance *= 1.0 - saturate(layers.surface.opacity);
        ray.TMin = realtimeNextLayer(layers.surface.t);
    }
}

// What a primary ray sees. Layers behind the nearest one coat the surface
// they lie on: their closures, shading normals and emission are blended into
// it by coverage, so ReSTIR and NRD light them with it. Where nothing opaque
// follows, they keep only their emission.
struct PrimaryVisibility
{
    // The nearest fractional layer, where `opacity` is positive.
    RealtimeHitPayload front;
    float opacity;
    // The first opaque surface with the layers behind `front` blended in, or
    // the miss.
    RealtimeHitPayload surface;
    // Fraction of the surface's light, or the background's, that reaches the
    // camera.
    float transmittance;
    // Emission of the layers in front of the background, weighted by the
    // transmittance in front of them.
    float3 emission;
};

// Camera rays skip the back faces of one-sided materials, as Unreal's path
// tracer's do.
RealtimeHitPayload traceCameraHitPayload(RayDesc ray, VertexDifferential differential)
{
    return traceRealtimeHitPayload(ray, differential, RaytracingMaskCamera,
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES);
}

PrimaryVisibility tracePrimaryVisibility(RayDesc ray, VertexDifferential differential)
{
    PrimaryVisibility result;
    result.front = traceCameraHitPayload(ray, differential);
    result.surface = result.front;
    result.opacity = saturate(result.front.opacity);
    result.transmittance = 1.0;
    result.emission = float3(0.0);
    if (result.front.instance == RealtimeMissInstance || result.opacity >= 1.0)
    {
        result.opacity = 0.0;
        return result;
    }

    // Coverage-weighted sums over the layers behind the front one.
    RealtimeClosure closure = {};
    float3 normal = float3(0.0);
    float3 emission = float3(0.0);
    float transmittance = 1.0;
    ray.TMin = realtimeNextLayer(result.front.t);
    [loop]
    for (uint layer = 2u; ; ++layer)
    {
        result.surface = traceCameraHitPayload(ray, differential);
        if (result.surface.instance == RealtimeMissInstance)
        {
            result.transmittance = (1.0 - result.opacity) * transmittance;
            result.emission = (1.0 - result.opacity) * emission;
            return result;
        }
        const bool ends = realtimeLayerEndsRay(result.surface, layer, transmittance);
        const RealtimeHit hit = unpackRealtimeHit(result.surface);
        const float coverage = transmittance * (ends ? 1.0 : saturate(hit.opacity));
        closure.diffuse += coverage * hit.closure.diffuse;
        closure.roughness += coverage * hit.closure.roughness;
        closure.specularF0 += coverage * hit.closure.specularF0;
        closure.specularF90 += coverage * hit.closure.specularF90;
        normal += coverage * realtimeHitBsdf(hit, ray.Direction).normal;
        emission += coverage * max(hit.emission, 0.0);
        if (ends)
        {
            packRealtimeHit(result.surface, hit.instance, hit.t, hit.position,
                hit.geometricNormal, normalize(normal), closure, emission);
            result.transmittance = 1.0 - result.opacity;
            return result;
        }
        transmittance *= 1.0 - saturate(result.surface.opacity);
        ray.TMin = realtimeNextLayer(result.surface.t);
    }
}
