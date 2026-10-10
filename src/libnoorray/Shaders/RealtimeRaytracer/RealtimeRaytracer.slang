import noorrhi;

#include "RealtimeHitShading.slang"
#include "../Raytracer/Sampling.slang"
#include "../Raytracer/CameraRay.slang"
#include "NrdSignal.slang"
#include "RealtimeTransparency.slang"

// Primary visibility of the realtime renderer, whose lighting ReSTIR DI and
// ReSTIR PT resample afterwards (RtxdiPasses.slang). This file builds two
// ray-generation stages:
//
// - REALTIME_GBUFFER: primary visibility at render resolution, through each
//   pixel's sub-pixel jitter for FSR, when lighting runs below it. It writes
//   the first-hit AOVs, FSR's depth and motion, the emission and background
//   that bypass the denoiser, and the material factors and guides the
//   composite resolves the lighting with.
// - Otherwise the lighting pass, at lighting resolution: one primary ray per
//   lighting pixel, jittered by the same screen distance. It records the
//   surfaces ReSTIR DI and PT resample on, the first opaque one and the
//   nearest fractional layer, and writes NRD's guides for both, which their
//   shading passes complete with the radiance inputs: packed normal and
//   roughness, view Z and motion. At full lighting resolution it also writes
//   everything the G-buffer pass would.
//   RealtimeComposite.slang upsamples and recombines the denoised signals
//   into the beauty image.
//
// A ray's hit shaders are its material's (MaterialHit.slang): the closest hit
// returns the material's closures, and the any-hit stages apply empty
// coverage. Rays pass fractional layers to the first opaque surface
// (RealtimeTransparency.slang), which the G-buffer, RTXDI and NRD see. The
// nearest fractional layer is a second surface with its own ReSTIR
// reservoirs, NRD signal and guides.

RayDesc realtimeRay(float3 origin, float3 direction)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.1;
    ray.TMax = 1.0e30;
    return ray;
}

// UV offset from this frame's unjittered position of a world point to its
// position in the previous frame.
float2 screenMotion(float3 worldPosition)
{
    const float4 current = transformToClip(realtimeArgs()->view.worldToClip, worldPosition);
    const float4 previous = transformToClip(realtimeArgs()->view.previousWorldToClip, worldPosition);
    if (current.w <= 1.0e-6 || previous.w <= 1.0e-6)
        return float2(0.0);
    const float2 currentUv = float2(0.5, 0.5) + float2(0.5, -0.5) * current.xy / current.w;
    const float2 previousUv = float2(0.5, 0.5) + float2(0.5, -0.5) * previous.xy / previous.w;
    return previousUv - currentUv;
}

// What a primary ray leaves at render resolution: the first-hit AOVs, FSR's
// depth and motion, the radiance that bypasses the denoiser, and what the
// composite resolves and remodulates the lighting with. The material factors
// carry the camera ray's weight, so the denoised lighting does not. The lighting pass
// writes the guides of its own samples from the same record.
struct PrimaryGBuffer
{
    float4 emission;
    float viewZ;
    float4 normalRoughness;
    float3 diffuseFactor;
    float3 specularFactor;
    uint cryptomatte;
    float deviceDepth;
    float2 motion;
};

// A ray that hit nothing: the environment, or transparency. Background lies
// beyond NRD's denoising range with nothing to remodulate, at zero inverted
// depth and moving like a point at infinity.
PrimaryGBuffer backgroundGBuffer(Frame frame, CameraRay cameraRay)
{
    RealtimeArgs* args = realtimeArgs();
    PrimaryGBuffer g;
    g.emission = float4(0.0, 0.0, 0.0, 1.0);
    g.viewZ = args.denoiser.denoisingRange * 2.0;
    g.normalRoughness = NRD_FrontEnd_PackNormalAndRoughness(float3(0.0, 0.0, 1.0), 1.0, 0.0);
    g.diffuseFactor = float3(0.0);
    g.specularFactor = float3(0.0);
    g.cryptomatte = 0xFFFFFFFFu;
    g.deviceDepth = 0.0;
    g.motion = cameraRay.valid
        ? screenMotion(cameraRay.origin + cameraRay.direction * 1.0e6) : float2(0.0);
    // Analytic light shapes are never visible to rays; they only contribute
    // through the light samples taken at surfaces.
    if (cameraRay.valid && frame.environment != nullptr)
    {
        Environment* environments = frame.environment;
        const float3 environment = environmentRadianceRgb(environments[0],
            cameraRay.direction, true);
        if (frame.transparentBackground == 0u)
            g.emission.rgb = cameraRay.weight * environment;
        else
            g.emission.a = 0.0;
    }
    return g;
}

PrimaryGBuffer surfaceGBuffer(CameraRay cameraRay, RealtimeBsdf bsdf, float3 position,
    uint instance, float3 emission)
{
    RealtimeArgs* args = realtimeArgs();
    PrimaryGBuffer g;
    g.emission = float4(cameraRay.weight * emission, 1.0);
    g.viewZ = dot(position - args.view.cameraPosition, args.view.cameraForward);
    g.normalRoughness = NRD_FrontEnd_PackNormalAndRoughness(bsdf.normal, bsdf.roughness, 0.0);
    NRD_MaterialFactors(bsdf.normal, bsdf.view, bsdf.diffuse, bsdf.specularF0, bsdf.roughness,
        g.diffuseFactor, g.specularFactor);
    g.diffuseFactor *= cameraRay.weight;
    g.specularFactor *= cameraRay.weight;
    g.cryptomatte = instance;
    g.deviceDepth = saturate(args.view.nearPlane / max(g.viewZ, args.view.nearPlane));
    g.motion = screenMotion(position);
    return g;
}

// Puts the layers a primary ray passed in front of its surface: they dim the
// surface's emission and lighting, and the transparent background's alpha,
// by their transmittance. The lighting stays demodulated without it, so the
// denoiser sees the surface alone.
void applyTranslucentLayers(inout PrimaryGBuffer g, PrimaryVisibility visibility,
    float cameraWeight)
{
    g.emission.rgb = g.emission.rgb * visibility.transmittance
        + cameraWeight * visibility.emission;
    g.emission.a = 1.0 - visibility.transmittance * (1.0 - g.emission.a);
    g.diffuseFactor *= visibility.transmittance;
    g.specularFactor *= visibility.transmittance;
}

void storeGBuffer(uint2 pixel, PrimaryGBuffer g)
{
    const RenderTargetHandles targets = realtimeArgs()->targets;
    RWTexture2D<uint> cryptomatte = ResourceDescriptorHeap[targets.cryptomatte];
    RWTexture2D<float4> depth = ResourceDescriptorHeap[targets.depth];
    RWTexture2D<float4> motion = ResourceDescriptorHeap[targets.motion];
    RWTexture2D<float4> normalRoughness = ResourceDescriptorHeap[targets.normalRoughness];
    RWTexture2D<float4> viewZ = ResourceDescriptorHeap[targets.viewZ];
    RWTexture2D<float4> emission = ResourceDescriptorHeap[targets.emission];
    RWTexture2D<float4> diffuseFactor = ResourceDescriptorHeap[targets.diffuseFactor];
    RWTexture2D<float4> specularFactor = ResourceDescriptorHeap[targets.specularFactor];
    cryptomatte[pixel] = g.cryptomatte;
    depth[pixel] = float4(g.deviceDepth, 0.0, 0.0, 0.0);
    motion[pixel] = float4(g.motion, 0.0, 0.0);
    normalRoughness[pixel] = g.normalRoughness;
    viewZ[pixel] = float4(g.viewZ, 0.0, 0.0, 0.0);
    emission[pixel] = float4(min(g.emission.rgb * realtimeExposureScale(), RealtimeHalfMax),
        g.emission.a);
    diffuseFactor[pixel] = float4(g.diffuseFactor, 0.0);
    specularFactor[pixel] = float4(g.specularFactor, 0.0);
}

void storeLayerGBuffer(uint2 pixel, PrimaryVisibility visibility, CameraRay cameraRay,
    inout PrimaryGBuffer base)
{
    const RenderTargetHandles targets = realtimeArgs()->targets;
    RWTexture2D<float4> diffuse = ResourceDescriptorHeap[targets.layerDiffuseFactor];
    RWTexture2D<float4> specular = ResourceDescriptorHeap[targets.layerSpecularFactor];
    RWTexture2D<float4> normal = ResourceDescriptorHeap[targets.layerNormalRoughness];
    RWTexture2D<float4> viewZ = ResourceDescriptorHeap[targets.layerViewZ];
    RWTexture2D<float4> mask = ResourceDescriptorHeap[targets.transparencyMask];
    const bool hasLayer = visibility.opacity > 0.0;
    mask[pixel] = float4(visibility.opacity, 0.0, 0.0, 0.0);
    if (!hasLayer)
    {
        diffuse[pixel] = float4(0.0);
        specular[pixel] = float4(0.0);
        normal[pixel] = NRD_FrontEnd_PackNormalAndRoughness(float3(0.0, 0.0, 1.0), 1.0, 0.0);
        viewZ[pixel] = float4(realtimeArgs()->denoiser.denoisingRange * 2.0, 0.0, 0.0, 0.0);
        return;
    }
    const RealtimeHit hit = unpackRealtimeHit(visibility.front);
    const PrimaryGBuffer layer = surfaceGBuffer(cameraRay,
        realtimeHitBsdf(hit, cameraRay.direction), hit.position, hit.instance,
        max(hit.emission, 0.0));
    diffuse[pixel] = float4(layer.diffuseFactor * visibility.opacity, 0.0);
    specular[pixel] = float4(layer.specularFactor * visibility.opacity, 0.0);
    normal[pixel] = layer.normalRoughness;
    viewZ[pixel] = float4(layer.viewZ, 0.0, 0.0, 0.0);
    base.emission.rgb += layer.emission.rgb * visibility.opacity;
    base.cryptomatte = layer.cryptomatte;
}

// The nearest fractional layer as a lighting sample: the surface its ReSTIR
// passes resample, and NRD's guides for its signal, which their shading pass
// writes.
RAB_Surface storeLayerLightingSample(uint2 pixel, PrimaryVisibility visibility,
    CameraRay cameraRay)
{
    const RenderTargetHandles targets = realtimeArgs()->targets;
    RWTexture2D<float4> normal = ResourceDescriptorHeap[targets.layerLightingNormalRoughness];
    RWTexture2D<float4> viewZ = ResourceDescriptorHeap[targets.layerLightingViewZ];
    RWTexture2D<float4> motion = ResourceDescriptorHeap[targets.layerLightingMotion];
    if (!(visibility.opacity > 0.0))
    {
        normal[pixel] = NRD_FrontEnd_PackNormalAndRoughness(float3(0.0, 0.0, 1.0), 1.0, 0.0);
        viewZ[pixel] = float4(realtimeArgs()->denoiser.denoisingRange * 2.0, 0.0, 0.0, 0.0);
        motion[pixel] = float4(0.0);
        return RAB_EmptySurface();
    }
    const RealtimeHit hit = unpackRealtimeHit(visibility.front);
    const RealtimeBsdf bsdf = realtimeHitBsdf(hit, cameraRay.direction);
    const float depth = dot(hit.position - realtimeArgs()->view.cameraPosition,
        realtimeArgs()->view.cameraForward);
    normal[pixel] = NRD_FrontEnd_PackNormalAndRoughness(bsdf.normal, bsdf.roughness, 0.0);
    viewZ[pixel] = float4(depth, 0.0, 0.0, 0.0);
    motion[pixel] = float4(screenMotion(hit.position), 0.0, 0.0);
    return makeRealtimeSurface(visibility.front, bsdf,
        max(depth, realtimeArgs()->view.nearPlane));
}


#ifndef REALTIME_LAYERS
#define REALTIME_LAYERS 0
#endif

#ifndef REALTIME_GBUFFER
#define REALTIME_GBUFFER 0
#endif

#if REALTIME_GBUFFER

[shader("raygeneration")]
void main()
{
    const Frame frame = realtimeArgs()->frame;
    const uint2 pixel = DispatchRaysIndex().xy;
    if (pixel.x >= frame.width || pixel.y >= frame.height)
        return;
    const float2 uv = (float2(pixel) + 0.5 + realtimeArgs()->view.jitter)
        / float2(frame.width, frame.height);
    const CameraRay cameraRay = generateCameraRay(frame, uv, float2(0.5), 550.0);
    if (!cameraRay.valid || frame.topLevelAS == 0)
    {
        PrimaryVisibility empty = {};
        PrimaryGBuffer background = backgroundGBuffer(frame, cameraRay);
#if REALTIME_LAYERS
        storeLayerGBuffer(pixel, empty, cameraRay, background);
#endif
        storeGBuffer(pixel, background);
        return;
    }
    const PrimaryVisibility visibility = tracePrimaryVisibility(
        realtimeRay(cameraRay.origin, cameraRay.direction),
        cameraRayDifferential(frame, cameraRay, uv, float2(0.5), 550.0));
    const RealtimeHit hit = unpackRealtimeHit(visibility.surface);
    PrimaryGBuffer g = hit.hit
        ? surfaceGBuffer(cameraRay, realtimeHitBsdf(hit, cameraRay.direction),
            hit.position, hit.instance, max(hit.emission, 0.0))
        : backgroundGBuffer(frame, cameraRay);
#if REALTIME_LAYERS
    applyTranslucentLayers(g, visibility, cameraRay.weight);
    storeLayerGBuffer(pixel, visibility, cameraRay, g);
#endif
    storeGBuffer(pixel, g);
}

#else // REALTIME_GBUFFER

[shader("raygeneration")]
void main()
{
    const Frame frame = realtimeArgs()->frame;
    RealtimeArgs* args = realtimeArgs();
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 lightingSize = uint2(args.view.lightingWidth, args.view.lightingHeight);
    if (any(pixel >= lightingSize))
        return;

    // One primary ray per lighting pixel: NRD and FSR accumulate over frames,
    // and the per-frame jitter spreads those rays across each pixel's
    // footprint.
    const float2 uv = (float2(pixel) + 0.5 + args.view.jitter / float(args.view.lightingScale))
        / float2(lightingSize);
    const CameraRay cameraRay = generateCameraRay(frame, uv, float2(0.5), 550.0);

    PrimaryVisibility visibility = {};
    visibility.surface.instance = RealtimeMissInstance;
    visibility.transmittance = 1.0;
    if (cameraRay.valid && frame.topLevelAS != 0)
    {
        visibility = tracePrimaryVisibility(realtimeRay(cameraRay.origin, cameraRay.direction),
            cameraRayDifferential(frame, cameraRay, uv, float2(0.5), 550.0));
    }
    const RealtimeHit hit = unpackRealtimeHit(visibility.surface);

    PrimaryGBuffer g;
    RAB_Surface surface = RAB_EmptySurface();
    if (!hit.hit)
        g = backgroundGBuffer(frame, cameraRay);
    else
    {
        const RealtimeBsdf bsdf = realtimeHitBsdf(hit, cameraRay.direction);
        g = surfaceGBuffer(cameraRay, bsdf, hit.position, hit.instance,
            max(hit.emission, 0.0));
        surface = makeRealtimeSurface(visibility.surface, bsdf,
            max(g.viewZ, args.view.nearPlane));
    }
    storeRealtimeSurface(args.lighting.surfaces, args.lighting.differentials,
        realtimeSurfaceIndex(pixel), surface);
#if REALTIME_LAYERS
    storeRealtimeSurface(args.lighting.layerSurfaces, args.lighting.layerDifferentials,
        realtimeSurfaceIndex(pixel), storeLayerLightingSample(pixel, visibility, cameraRay));
#endif

    const RenderTargetHandles targets = args.targets;
    RWTexture2D<float4> normalRoughnessTarget = ResourceDescriptorHeap[targets.lightingNormalRoughness];
    RWTexture2D<float4> viewZTarget = ResourceDescriptorHeap[targets.lightingViewZ];
    RWTexture2D<float4> motionTarget = ResourceDescriptorHeap[targets.lightingMotion];
    normalRoughnessTarget[pixel] = g.normalRoughness;
    viewZTarget[pixel] = float4(g.viewZ, 0.0, 0.0, 0.0);
    motionTarget[pixel] = float4(g.motion, 0.0, 0.0);
    // At full lighting resolution there is no separate G-buffer pass.
    if (args.view.lightingScale == 1.0f)
    {
#if REALTIME_LAYERS
        applyTranslucentLayers(g, visibility, cameraRay.weight);
        storeLayerGBuffer(pixel, visibility, cameraRay, g);
#endif
        storeGBuffer(pixel, g);
    }
}

#endif // REALTIME_GBUFFER
