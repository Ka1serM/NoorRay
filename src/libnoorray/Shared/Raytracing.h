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
    // The Gaussian splat instance the path last scattered from, which its
    // next rays must not hit again.
    uint splatInstance;
    float gaussianOpacity;
    float3 radiance;
    float3 albedo;
    float3 normal;
    float3 geometricNormal;
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
// where the ray landed and the closure its material produced there, resolved
// and packed as HitPayload.slang describes; the ray-generation stage does all
// lighting. Any-hit stages only read it.
struct RealtimeHitPayload
{
    // RealtimeMissInstance when the ray escaped.
    uint instance;
    float t;
    float3 position;
    uint geometricNormal;
    uint normal;
    float opacity;
    // Halves: diffuse, roughness, specular F0 and F90.
    uint closure[5];
    float3 emission;
    // A VertexDifferential (RayDifferential.slang) as halves: the ray's on the
    // way in, the hit's on the way out.
    uint differential[9];
    // RealtimeHit* bits.
    uint flags;
};

static const uint RealtimeMissInstance = 0xFFFFFFFFu;
// The hit's material is an Unreal "Is Sky" material (MaterialFlagSky).
static const uint RealtimeHitSky = 0x1u;

// Payload of the realtime renderer's visibility rays: they skip closest hits,
// so a ray is occluded unless the shadow miss stage clears `occluded`. The
// fractional surfaces it passes multiply `transmittance` in their any-hit.
struct RealtimeShadowPayload
{
    uint occluded;
    float transmittance;
};

// The realtime renderer's ray types: each is one shader-binding-table record
// per mesh section, selected by TraceRay()'s record offset.
static const uint RealtimeRayTypeSurface = 0u;
static const uint RealtimeRayTypeShadow = 1u;
static const uint RaytracingRayTypeCount = 2u;
// Miss records of the realtime pipeline.
static const uint RealtimeMissSurface = 0u;
static const uint RealtimeMissShadow = 1u;

// TLAS instance visibility masks. A mesh instance carries the ray kinds that
// find it (MeshInstance::RayVisibility), as Unreal's path tracer masks
// primitives: camera rays, shadow rays and the rays of later bounces. The
// spectral renderer traces meshes of any visibility, and Gaussian splats;
// the realtime renderer traces meshes only.
static const uint RaytracingMaskCamera = 0x01u;
static const uint RaytracingMaskGaussian = 0x02u;
static const uint RaytracingMaskShadow = 0x04u;
static const uint RaytracingMaskIndirect = 0x08u;
static const uint RaytracingMaskMesh = RaytracingMaskCamera | RaytracingMaskShadow | RaytracingMaskIndirect;

#ifdef __cplusplus
} // namespace nr::graphics
#endif
