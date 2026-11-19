#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/Shared.h"
#if defined(__SLANG__)
#define RTXDI_COLOR float4
#define RTXDI_COLOR_COMPONENTS 4
//! WGSL has no 16-bit integers; RTXDI keeps only small counters and one enum field in uint16_t.
#define uint16_t uint
#else
#include <cstdint>

#include "NoorRay/Shaders/Wavefront/WavefrontRecords.h"
#endif
#include "Rtxdi/PT/ReSTIRPTParameters.h"

NR_NAMESPACE_BEGIN

static const uint restirGroupSize = 64u;

//! normal is the octahedral geometric normal on the side of the incoming ray; depth is 0 for a pixel without a surface.
struct RestirSurfaceRecord {
    float3 position;
    float depth;
    uint normal;
    uint view;
    uint material;
    uint cone;
    uint instance;
    uint primitive;
    float2 barycentrics;
};

struct RestirFrame {
    Ptr<WavefrontFrame> wavefront;
    Ptr<RTXDI_PackedPTReservoir> reservoirs;
    Ptr<float2> neighborOffsets;
    Ptr<RestirSurfaceRecord> surfaces;
    Ptr<RestirSurfaceRecord> previousSurfaces;
    uint width;
    uint height;
    uint seed;
    uint gridThreads;
    RTXDI_RuntimeParameters runtime;
    RTXDI_PTParameters pt;
};

struct RestirArguments {
    Ptr<RestirFrame> frame;
};

NR_NAMESPACE_END

#if !defined(__SLANG__)
#include <cstddef>
static_assert(sizeof(RTXDI_PackedPTReservoir) == 80);
static_assert(sizeof(RTXDI_PTParameters) == 240);
static_assert(sizeof(NoorRay::Shaders::RestirSurfaceRecord) == 48);
static_assert(offsetof(NoorRay::Shaders::RestirFrame, runtime) == 56);
static_assert(offsetof(NoorRay::Shaders::RestirFrame, pt) == 72);
static_assert(sizeof(NoorRay::Shaders::RestirFrame) == 312);
static_assert(offsetof(RTXDI_PTReconnectionParameters, pad1) == offsetof(RTXDI_PTReconnectionParameters, reconnectionMode) + 4);
#endif
