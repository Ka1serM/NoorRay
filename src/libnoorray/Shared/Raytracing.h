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

// Payload of the RGB preview renderer. Every radiometric value is linear
// Rec.709 RGB; the hit shader returns one resolved scattering event.
struct RealtimeRayPayload
{
    uint hit;
    uint instance;
    float t;
    uint seed;
    uint depth;
    // Regularizes paths after glossy scattering.
    float roughnessFloor;
    // Set by the caller when ReSTIR DI lights this hit's analytic lights in
    // screen space, so the hit shader samples only the environment.
    uint screenSpaceLights;
    float3 position;
    float3 normal;
    float3 geometricNormal;
    float3 albedo;
    float3 emission;
    // Material terms the denoiser demodulates by.
    float3 diffuseAlbedo;
    float3 specularF0;
    float3 specularF90;
    float roughness;
    // Unoccluded light-sample contribution, split by lobe.
    float3 directDiffuse;
    float3 directSpecular;
    float3 shadowDirection;
    float shadowDistance;
    uint shadowValid;
    // A second, independently shadowed light sample: the analytic lights
    // resampled through RTXDI's world-space RIS when RTXDI owns them (the
    // sample above is then the environment's alone). Primary hits leave it
    // empty; ReSTIR DI lights them in screen space instead.
    float3 lightDiffuse;
    float3 lightSpecular;
    float3 lightDirection;
    float lightDistance;
    uint lightValid;
    float3 bsdfWeight;
    float3 nextDirection;
    uint sampleValid;
    // Set when the continuation came from the specular lobe.
    uint sampledSpecular;
    // Solid-angle pdf of nextDirection for multiple importance sampling
    // against light sampling.
    float bsdfPdf;
    float sampledRoughness;
};

// TLAS instance visibility masks. The spectral renderer traces both; the
// realtime renderer traces meshes only.
static const uint RaytracingMaskMesh = 0x01u;
static const uint RaytracingMaskGaussian = 0x02u;

#ifdef __cplusplus
} // namespace nr::graphics
#endif
