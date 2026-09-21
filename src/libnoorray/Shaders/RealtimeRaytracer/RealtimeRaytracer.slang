#include "RtxdiBridge.slang"
#include "Sharc.slang"
#include "../Raytracer/Sampling.slang"
#include "../Raytracer/CameraRay.slang"
#include "NrdSignal.slang"
#include "Rtxdi/GI/Reservoir.hlsli"

// RGB preview path tracer with SHaRC global illumination. This file builds
// two ray-generation stages:
//
// - SHARC_UPDATE: a sparse pass tracing one path per sharcUpdateStride^2
//   pixel block. Every vertex reports its emission and shadowed direct light
//   to the cache, which propagates it back along the path.
// - SHARC_QUERY: the image pass, at render resolution. One path per pixel,
//   through the pixel's sub-pixel jitter for FSR. It writes the first-hit
//   AOVs, FSR's depth and motion, and NRD's REBLUR inputs: demodulated
//   diffuse and specular radiance with the first-bounce hit distance, packed
//   normal and roughness, view Z and world-space motion. Emission and
//   background bypass the denoiser. RealtimeComposite.slang recombines the
//   denoised signals into the beauty image. After a rough scattering event a
//   path ends at the first cached voxel it reaches.
//
// Light transport is deliberately cheaper than the reference: every lobe
// smoother than earlier path vertices widened to their roughness, and only
// mesh instances traced, all as opaque geometry. Analytic lights are sampled
// through RTXDI (RtxdiBridge.slang): primary surfaces by ReSTIR DI in the
// passes of RtxdiPasses.slang, every later vertex by RIS over RTXDI's ReGIR
// world-space light grid. The environment keeps its own light sample with MIS
// against BSDF sampling, and analytic light shapes are reached by light
// sampling only. Without RTXDI resources (enabled == 0) each scattering event
// instead takes one light sample over all lights, and a bounce may land on a
// light shape, MIS-weighted against it.

// Paths are terminated in the cache only after an event at least this rough;
// sharper reflections trace on so mirrors do not show voxel-sized radiance.
static const float SharcQueryRoughness = 0.25;
static const uint SharcUpdateMaxBounces = 4u;
static const uint RealtimeRayFlags = RAY_FLAG_FORCE_OPAQUE;

float3 offsetRayOrigin(float3 position, float3 normal, float3 direction)
{
    return position + normal * (dot(normal, direction) >= 0.0 ? 1.0e-4 : -1.0e-4);
}

// Shadowed contribution of the light samples a hit shader returned, split by
// lobe: the (environment or all-light) sample, then RTXDI's analytic one.
struct VertexDirectLight
{
    float3 diffuse;
    float3 specular;
};

VertexDirectLight shadowedDirectLight(RealtimeRayPayload payload)
{
    VertexDirectLight result;
    result.diffuse = float3(0.0);
    result.specular = float3(0.0);
    if (payload.shadowValid != 0u && realtimeVisible(payload.position,
            payload.geometricNormal, payload.shadowDirection, payload.shadowDistance))
    {
        result.diffuse += payload.directDiffuse;
        result.specular += payload.directSpecular;
    }
    if (payload.lightValid != 0u && realtimeVisible(payload.position,
            payload.geometricNormal, payload.lightDirection, payload.lightDistance))
    {
        result.diffuse += payload.lightDiffuse;
        result.specular += payload.lightSpecular;
    }
    return result;
}

RealtimeRayPayload traceRealtimeRay(Frame frame, float3 origin,
    float3 direction, uint depth, uint seed, float roughnessFloor,
    bool screenSpaceLights = false)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001;
    ray.TMax = 1.0e30;
    RealtimeRayPayload payload = {};
    payload.seed = seed;
    payload.depth = depth;
    payload.roughnessFloor = roughnessFloor;
    payload.screenSpaceLights = screenSpaceLights ? 1u : 0u;
    TraceRay(RaytracingAccelerationStructure(frame.topLevelAS),
        RealtimeRayFlags, RaytracingMaskMesh, 0, 2, 0, ray, payload);
    return payload;
}

SharcHitData makeSharcHit(RealtimeRayPayload payload)
{
    SharcHitData hit;
    hit.positionWorld = payload.position;
    hit.normalWorld = payload.geometricNormal;
    return hit;
}

#if SHARC_UPDATE

[shader("raygeneration")]
void main()
{
    const Frame frame = realtimeArgs()->frame;
    if (!sharcEnabled() || frame.topLevelAS == 0)
        return;
    const uint3 dispatchId = DispatchRaysIndex();
    const uint2 grid = uint2(max(realtimeArgs()->radianceCache.updateGridWidth, 1u),
        max(realtimeArgs()->radianceCache.updateGridHeight, 1u));
    if (dispatchId.x >= grid.x || dispatchId.y >= grid.y)
        return;

    // One path per grid cell, landing anywhere inside it, so the cache sees
    // the whole view over a few frames. The grid is the view, not the render
    // target: the voxels being filled are sized from camera distance, so what
    // this pass has to cover is the field of view and not however many pixels
    // that view is currently traced with.
    Rng pathRng;
    pathRng.state = pcgHash(hashCombine32(realtimeArgs()->view.frameIndex,
        dispatchId.y * 65521u + dispatchId.x) ^ 0x53484152u);
    const float2 uv = (float2(dispatchId.xy) + pathRng.nextFloat2()) / float2(grid);
    const CameraRay cameraRay = generateCameraRay(frame, uv, float2(0.5), 550.0);
    if (!cameraRay.valid)
        return;

    const SharcParameters sharc = makeSharcParameters();
    SharcState sharcState;
    SharcInit(sharcState);

    float3 rayOrigin = cameraRay.origin;
    float3 rayDirection = cameraRay.direction;
    float lastBsdfPdf = 0.0;
    float roughnessFloor = 0.0;
    const bool rtxdi = realtimeRtxdiEnabled();
    // Light shapes RTXDI samples are not also found by BSDF sampling.
    const bool analyticLights = hasAnalyticLights(frame) && !rtxdi;
    const float environmentSelection = realtimeEnvironmentSelection(frame, rtxdi);
    const uint bounceCount = min(max(frame.maxBounces, 1u), SharcUpdateMaxBounces);
    [loop]
    for (uint depth = 0u; depth < bounceCount; ++depth)
    {
        const RealtimeRayPayload payload = traceRealtimeRay(frame, rayOrigin,
            rayDirection, depth, pathRng.state, roughnessFloor);
        pathRng.state = payload.seed;

        // Light shapes stay invisible to the camera ray this pass starts
        // from (depth 0), but a bounce that lands on one behaves like any
        // other emitter, MIS-weighted against the light sampling already
        // done at the previous vertex.
        if (analyticLights && depth != 0u)
        {
            const float surfaceDistance = payload.hit == RealtimeMiss
                ? 1.0e30 : payload.t;
            const AnalyticLightHitRgb lightHit = intersectAnalyticLightsRgb(
                makeLightTable(frame), rayOrigin, rayDirection, 0.001,
                surfaceDistance);
            if (lightHit.valid && (payload.hit == RealtimeMiss
                    || lightHit.distance < surfaceDistance))
            {
                float misWeight = 1.0;
                if (lastBsdfPdf > 0.0 && lightHit.pdf > 0.0)
                    misWeight = powerHeuristic(lastBsdfPdf,
                        (1.0 - environmentSelection) * lightHit.pdf);
                SharcUpdateMiss(sharc, sharcState, lightHit.radiance * misWeight);
                break;
            }
        }

        if (payload.hit == RealtimeMiss)
        {
            if (frame.environment != nullptr)
            {
                Environment* environments = frame.environment;
                float misWeight = 1.0;
                if (lastBsdfPdf > 0.0 && environmentSelection > 0.0)
                    misWeight = powerHeuristic(lastBsdfPdf, environmentSelection
                        * environmentPdf(environments[0], rayDirection));
                SharcUpdateMiss(sharc, sharcState, environmentRadianceRgb(
                    environments[0], rayDirection, false) * misWeight);
            }
            break;
        }

        const VertexDirectLight direct = shadowedDirectLight(payload);
        const float3 vertexRadiance = payload.emission + direct.diffuse + direct.specular;
        if (!SharcUpdateHit(sharc, sharcState, makeSharcHit(payload),
                vertexRadiance, pathRng.nextFloat()))
            break;

        if (payload.sampleValid == 0u || !realtimeIsFinite(payload.bsdfWeight)
            || realtimeMaxComponent(payload.bsdfWeight) <= 0.0)
            break;
        // SHaRC carries path throughput per segment.
        SharcSetThroughput(sharcState, payload.bsdfWeight);
        lastBsdfPdf = payload.bsdfPdf;
        roughnessFloor = max(roughnessFloor, payload.sampledRoughness);
        rayDirection = payload.nextDirection;
        rayOrigin = offsetRayOrigin(payload.position, payload.geometricNormal,
            rayDirection);
    }
}

#else // SHARC_UPDATE

struct IndirectPath
{
    float3 radiance;
    // First-bounce segment length for NRD; misses count as unoccluded.
    float hitDistance;
    // Where the continuation ray first landed, for ReSTIR GI: the secondary
    // surface, or a far point along the ray when it escaped.
    float3 secondaryPosition;
    float3 secondaryNormal;
};

// Radiance arriving at a primary hit along its sampled continuation, before
// the primary BSDF weight is applied.
IndirectPath traceIndirect(Frame frame, RealtimeRayPayload primary, inout Rng pathRng)
{
    IndirectPath result;
    result.radiance = float3(0.0);
    result.hitDistance = NRD_INF;

    const SharcParameters sharc = makeSharcParameters();
    const bool useCache = sharcEnabled();
    const bool rtxdi = realtimeRtxdiEnabled();
    const bool analyticLights = hasAnalyticLights(frame) && !rtxdi;
    const float environmentSelection = realtimeEnvironmentSelection(frame, rtxdi);
    float3 rayDirection = primary.nextDirection;
    float3 rayOrigin = offsetRayOrigin(primary.position, primary.geometricNormal,
        rayDirection);
    result.secondaryPosition = rayOrigin + rayDirection * realtimeArgs()->denoiser.denoisingRange;
    result.secondaryNormal = -rayDirection;
    float3 throughput = float3(1.0);
    float lastBsdfPdf = primary.bsdfPdf;
    float lastRoughness = primary.sampledRoughness;
    float roughnessFloor = primary.sampledRoughness;
    const uint bounceCount = max(frame.maxBounces, 1u);
    [loop]
    for (uint depth = 1u; depth < bounceCount; ++depth)
    {
        const RealtimeRayPayload payload = traceRealtimeRay(frame, rayOrigin,
            rayDirection, depth, pathRng.state, roughnessFloor);
        pathRng.state = payload.seed;

        // Unlike the primary camera ray, a reflected or GI ray that lands on
        // a light shape does see it: MIS-weighted against the light sample
        // already taken at the previous vertex, same as any other emitter.
        if (analyticLights)
        {
            const float surfaceDistance = payload.hit == RealtimeMiss
                ? 1.0e30 : payload.t;
            const AnalyticLightHitRgb lightHit = intersectAnalyticLightsRgb(
                makeLightTable(frame), rayOrigin, rayDirection, 0.001,
                surfaceDistance);
            if (lightHit.valid && (payload.hit == RealtimeMiss
                    || lightHit.distance < surfaceDistance))
            {
                float misWeight = 1.0;
                if (lastBsdfPdf > 0.0 && lightHit.pdf > 0.0)
                    misWeight = powerHeuristic(lastBsdfPdf,
                        (1.0 - environmentSelection) * lightHit.pdf);
                result.radiance += clampIndirectRgb(throughput * lightHit.radiance
                    * misWeight, frame.indirectLightClamp, depth);
                if (depth == 1u)
                    result.hitDistance = lightHit.distance;
                break;
            }
        }

        if (depth == 1u && payload.hit != RealtimeMiss)
        {
            result.hitDistance = payload.t;
            result.secondaryPosition = payload.position;
            result.secondaryNormal = payload.geometricNormal;
        }

        if (payload.hit == RealtimeMiss)
        {
            if (frame.environment != nullptr)
            {
                Environment* environments = frame.environment;
                float misWeight = 1.0;
                if (lastBsdfPdf > 0.0 && environmentSelection > 0.0)
                    misWeight = powerHeuristic(lastBsdfPdf, environmentSelection
                        * environmentPdf(environments[0], rayDirection));
                result.radiance += clampIndirectRgb(throughput * environmentRadianceRgb(
                    environments[0], rayDirection, false) * misWeight,
                    frame.indirectLightClamp, depth);
            }
            break;
        }

        // The cached radiance already holds this vertex's emission, direct
        // light and everything scattered beyond it. Hits closer than one
        // voxel keep tracing so light does not leak around contact corners.
        if (useCache && lastRoughness >= SharcQueryRoughness)
        {
            const float voxelSize = HashGridGetVoxelSize(HashGridGetLevel(
                payload.position, sharc.hashGridParameters),
                sharc.hashGridParameters);
            float3 cached = float3(0.0);
            if (payload.t > voxelSize
                && SharcGetCachedRadiance(sharc, makeSharcHit(payload), cached, false))
            {
                result.radiance += clampIndirectRgb(throughput * cached,
                    frame.indirectLightClamp, depth);
                break;
            }
        }

        result.radiance += clampIndirectRgb(throughput * payload.emission,
            frame.indirectLightClamp, depth);
        const VertexDirectLight direct = shadowedDirectLight(payload);
        result.radiance += clampIndirectRgb(throughput * (direct.diffuse + direct.specular),
            frame.indirectLightClamp, depth);

        if (payload.sampleValid == 0u)
            break;
        throughput *= payload.bsdfWeight;
        lastBsdfPdf = payload.bsdfPdf;
        lastRoughness = payload.sampledRoughness;
        roughnessFloor = max(roughnessFloor, payload.sampledRoughness);
        if (!realtimeIsFinite(throughput) || realtimeMaxComponent(throughput) <= 0.0)
            break;
        if (depth >= 2u)
        {
            const float survival = clamp(realtimeMaxComponent(throughput),
                0.05, 0.95);
            if (pathRng.nextFloat() >= survival)
                break;
            throughput /= survival;
        }
        rayDirection = payload.nextDirection;
        rayOrigin = offsetRayOrigin(payload.position, payload.geometricNormal,
            rayDirection);
    }
    if (!realtimeIsFinite(result.radiance))
        result.radiance = float3(0.0);
    return result;
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

[shader("raygeneration")]
void main()
{
    const Frame frame = realtimeArgs()->frame;
    RealtimeArgs* args = realtimeArgs();
    const uint3 dispatchId = DispatchRaysIndex();
    if (dispatchId.x >= frame.width || dispatchId.y >= frame.height)
        return;
    const uint2 pixel = dispatchId.xy;

    RWTexture2D<float4> albedo = ResourceDescriptorHeap[args.targets.albedo];
    RWTexture2D<float4> normal = ResourceDescriptorHeap[args.targets.normal];
    RWTexture2D<float4> position = ResourceDescriptorHeap[args.targets.position];
    RWTexture2D<uint> cryptomatte = ResourceDescriptorHeap[args.targets.cryptomatte];
    RWTexture2D<float4> depthTarget = ResourceDescriptorHeap[args.targets.depth];
    RWTexture2D<float4> motionTarget = ResourceDescriptorHeap[args.targets.motion];
    RWTexture2D<float4> diffuseTarget = ResourceDescriptorHeap[args.targets.diffuse];
    RWTexture2D<float4> specularTarget = ResourceDescriptorHeap[args.targets.specular];
    RWTexture2D<float4> normalRoughnessTarget = ResourceDescriptorHeap[args.targets.normalRoughness];
    RWTexture2D<float4> viewZTarget = ResourceDescriptorHeap[args.targets.viewZ];
    RWTexture2D<float4> emissionTarget = ResourceDescriptorHeap[args.targets.emission];
    RWTexture2D<float4> diffuseFactorTarget = ResourceDescriptorHeap[args.targets.diffuseFactor];
    RWTexture2D<float4> specularFactorTarget = ResourceDescriptorHeap[args.targets.specularFactor];

    // One path per pixel: NRD and FSR accumulate over frames, and the
    // per-frame jitter spreads those paths across each pixel's footprint.
    Rng pathRng;
    pathRng.state = pcgHash(hashCombine32(args.view.frameIndex,
        dispatchId.y * frame.width + dispatchId.x));
    const float2 uv = (float2(pixel) + 0.5 + args.view.jitter)
        / float2(frame.width, frame.height);
    const CameraRay cameraRay = generateCameraRay(frame, uv, float2(0.5), 550.0);

    // Background defaults: beyond NRD's denoising range, nothing to remodulate.
    float4 emission = float4(0.0, 0.0, 0.0, 1.0);
    float viewZ = args.denoiser.denoisingRange * 2.0;
    float4 normalRoughness = NRD_FrontEnd_PackNormalAndRoughness(
        float3(0.0, 0.0, 1.0), 1.0, 0.0);
    float4 diffuseSignal = float4(0.0);
    float4 specularSignal = float4(0.0);
    float3 diffuseFactor = float3(0.0);
    float3 specularFactor = float3(0.0);
    float3 firstAlbedo = float3(0.0);
    float3 firstNormal = float3(0.0, 0.0, 1.0);
    float3 firstPosition = float3(0.0);
    uint firstCryptomatte = 0xFFFFFFFFu;
    // Background: zero inverted depth, motion of a point at infinity.
    float deviceDepth = 0.0;
    float2 motion = cameraRay.valid
        ? screenMotion(cameraRay.origin + cameraRay.direction * 1.0e6) : float2(0.0);

    // With RTXDI, the primary hit's analytic lights are resampled by the
    // ReSTIR DI passes that follow this one; this pass then leaves unpacked
    // radiance in NRD's inputs for them to add to and pack, and records the
    // surface they resample on. In ReSTIR GI mode the diffuse continuation
    // becomes that pass's initial sample instead of being added here.
    const bool rtxdi = realtimeRtxdiEnabled();
    RealtimeLighting* lighting = &args.lighting;
    const bool restirGI = rtxdi && lighting.mode == RealtimeLightingReSTIRGI;
    RAB_Surface gbufferSurface = RAB_EmptySurface();
    RTXDI_GIReservoir giSample = RTXDI_EmptyGIReservoir();

    RealtimeRayPayload payload = {};
    payload.hit = RealtimeMiss;
    payload.t = 1.0e30;
    if (cameraRay.valid && frame.topLevelAS != 0)
        payload = traceRealtimeRay(frame, cameraRay.origin, cameraRay.direction,
            0u, pathRng.state, 0.0, rtxdi);
    pathRng.state = payload.seed;

    // Analytic light shapes are never visible to rays; they only contribute
    // through the light samples taken at surfaces.
    if (payload.hit == RealtimeMiss)
    {
        if (cameraRay.valid && frame.environment != nullptr)
        {
            Environment* environments = frame.environment;
            const float3 environment = environmentRadianceRgb(environments[0],
                cameraRay.direction, true);
            if (frame.transparentBackground == 0u)
                emission.rgb = cameraRay.weight * environment;
            else
                emission.a = 0.0;
            firstAlbedo = environment;
        }
    }
    else
    {
        firstAlbedo = payload.albedo;
        firstNormal = payload.normal;
        firstPosition = payload.position;
        firstCryptomatte = payload.instance;

        const float3 view = -cameraRay.direction;
        viewZ = dot(payload.position - args.view.cameraPosition, args.view.cameraForward);
        deviceDepth = saturate(args.view.nearPlane / max(viewZ, args.view.nearPlane));
        motion = screenMotion(payload.position);
        normalRoughness = NRD_FrontEnd_PackNormalAndRoughness(payload.normal,
            payload.roughness, 0.0);
        NRD_MaterialFactors(payload.normal, view, payload.diffuseAlbedo,
            payload.specularF0, payload.roughness, diffuseFactor, specularFactor);
        emission.rgb = cameraRay.weight * payload.emission;

        // The BSDF the hit shader sampled, rebuilt for the resampling passes.
        RealtimeBsdf bsdf;
        bsdf.normal = payload.normal;
        bsdf.geometricNormal = payload.geometricNormal;
        bsdf.view = view;
        bsdf.normalView = dot(payload.normal, view);
        bsdf.diffuse = payload.diffuseAlbedo;
        bsdf.specularF0 = payload.specularF0;
        bsdf.specularF90 = payload.specularF90;
        bsdf.roughness = payload.roughness;
        bsdf.diffuseProbability = realtimePrimaryDiffuseProbability(realtimeDiffuseProbability(
            bsdf.diffuse, bsdf.specularF0, bsdf.specularF90, bsdf.normalView));
        gbufferSurface = makeRealtimeSurface(bsdf, payload.position,
            max(viewZ, args.view.nearPlane));

        // A pixel whose diffuse lobe was not sampled still took one GI
        // sample, of zero radiance. Leaving its reservoir empty (M = 0) drops
        // it from the resampling normalization and brightens GI by roughly
        // 1 / diffuseProbability. The position is finite so reuse from a
        // neighbour computes a finite, zero-weighted Jacobian.
        if (restirGI)
            giSample = RTXDI_MakeGIReservoir(payload.position + payload.geometricNormal,
                -payload.geometricNormal, float3(0.0), 0.0);

        const VertexDirectLight direct = shadowedDirectLight(payload);
        float3 diffuseRadiance = direct.diffuse;
        float3 specularRadiance = direct.specular;
        float diffuseHitDistance = 0.0;
        float specularHitDistance = 0.0;
        if (payload.sampleValid != 0u)
        {
            const IndirectPath indirect = traceIndirect(frame, payload, pathRng);
            const float3 contribution = payload.bsdfWeight * indirect.radiance;
            if (payload.sampledSpecular != 0u)
            {
                specularRadiance += contribution;
                specularHitDistance = indirect.hitDistance;
            }
            else if (restirGI)
            {
                // The diffuse lobe was picked with probability
                // diffuseProbability and sampled by cosine.
                const float sourcePdf = bsdf.diffuseProbability
                    * bsdf.diffusePdf(payload.nextDirection);
                if (sourcePdf > 0.0 && realtimeIsFinite(indirect.radiance))
                    giSample = RTXDI_MakeGIReservoir(indirect.secondaryPosition,
                        indirect.secondaryNormal, indirect.radiance, sourcePdf);
            }
            else
            {
                diffuseRadiance += contribution;
                diffuseHitDistance = indirect.hitDistance;
            }
        }
        diffuseRadiance *= cameraRay.weight;
        specularRadiance *= cameraRay.weight;

        if (rtxdi)
        {
            diffuseSignal = float4(diffuseRadiance, diffuseHitDistance);
            specularSignal = float4(specularRadiance, specularHitDistance);
        }
        else
        {
            diffuseSignal = packNrdSignal(diffuseRadiance / diffuseFactor,
                diffuseHitDistance, viewZ, 1.0);
            specularSignal = packNrdSignal(specularRadiance / specularFactor,
                specularHitDistance, viewZ, payload.roughness);
        }
    }

    if (rtxdi)
    {
        lighting.surfaces[realtimeSurfaceIndex(pixel)] = packRealtimeSurface(gbufferSurface,
            cameraRay.weight);
        if (restirGI)
            RTXDI_StoreGIReservoir(giSample, lighting.restirGI.reservoirBufferParams, pixel,
                lighting.restirGI.bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex);
    }

    albedo[pixel] = float4(firstAlbedo, 1.0);
    normal[pixel] = float4(firstNormal, 1.0);
    position[pixel] = float4(firstPosition, 1.0);
    cryptomatte[pixel] = firstCryptomatte;
    depthTarget[pixel] = float4(deviceDepth, 0.0, 0.0, 0.0);
    motionTarget[pixel] = float4(motion, 0.0, 0.0);
    diffuseTarget[pixel] = diffuseSignal;
    specularTarget[pixel] = specularSignal;
    normalRoughnessTarget[pixel] = normalRoughness;
    viewZTarget[pixel] = float4(viewZ, 0.0, 0.0, 0.0);
    emissionTarget[pixel] = emission;
    diffuseFactorTarget[pixel] = float4(diffuseFactor, 0.0);
    specularFactorTarget[pixel] = float4(specularFactor, 0.0);
}

#endif // SHARC_UPDATE
