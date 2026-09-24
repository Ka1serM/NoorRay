#pragma once

#include "Frame.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

// Ray-tracing pipeline interface records. These declarations are consumed by
// both the host pipeline setup and the Slang stages; keeping them here avoids
// a second ABI definition in shader source.
struct RaytracerRoot
{
    GpuPtr(Frame) arguments;
};

struct RayPayload
{
    uint hit;
    uint instance;
    uint primitive;
    float2 barycentrics;
    float t;
    uint seed;
    uint gaussianId;
    float gaussianOpacity;
    float3 radiance;
    float3 albedo;
    float3 normal;
    float3 position;
    float alpha;
    uint cryptomatte;
    float4 bsdfWeight;
    float4 emissionSpectrum;
    float3 nextDirection;
    uint sampleValid;
    float4 directSpectrum;
    float3 shadowDirection;
    float shadowDistance;
    uint shadowValid;
    uint shadowInstance;
    uint shadowPrimitive;
    float wavelengthSample;
    uint depth;
    float bsdfPdf;
    float lightPdf;
};

// Payload of the realtime renderer's rays. A material's closest hit reports
// where the ray landed and the closures its material produced there
// (MaterialInterface.slang's BSDF, member by member); the ray-generation
// stage does all lighting. Any-hit stages leave it untouched.
struct RealtimeHitPayload
{
    uint hit;
    uint instance;
    float t;
    float3 position;
    float3 geometricNormal;
    float3 shadingNormal;
    float3 diffuse;
    float3 specularF0;
    float3 specularF90;
    float roughnessSum;
    float roughnessWeight;
    float3 normalSum;
    float3 emission;
};

// Payload of the realtime renderer's visibility rays: they skip closest hits,
// so a ray is occluded unless the shadow miss stage clears this.
struct RealtimeShadowPayload
{
    uint occluded;
};

// The realtime renderer's ray types: each is one shader-binding-table record
// per mesh section, selected by TraceRay()'s record offset.
static const uint RealtimeRayTypeSurface = 0u;
static const uint RealtimeRayTypeShadow = 1u;
static const uint RaytracingRayTypeCount = 2u;
// Miss records of the realtime pipeline.
static const uint RealtimeMissSurface = 0u;
static const uint RealtimeMissShadow = 1u;

// TLAS instance visibility masks. The spectral renderer traces both; the
// realtime renderer traces meshes only.
static const uint RaytracingMaskMesh = 0x01u;
static const uint RaytracingMaskGaussian = 0x02u;

#ifdef __cplusplus
} // namespace nr::graphics
#endif
