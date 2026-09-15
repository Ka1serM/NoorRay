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

#ifdef __cplusplus
} // namespace nr::graphics
#endif
