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
#include "Rtxdi/DI/ReSTIRDIParameters.h"
#include "Rtxdi/PT/ReSTIRPTParameters.h"

NR_NAMESPACE_BEGIN

static const uint restirGroupSize = 64u;

static const uint restirModeTemporal = 0u;
static const uint restirModeSpatial = 1u;
//! The resampling of a pixel runs in three stages: the forward shifts emit their replays, then the inverse shifts, then the pixel stores its reservoir.
static const uint restirStages = 3u;

//! The modes of the path kernels: the initial paths, the replays, and the end of the initial paths that are still in flight.
static const uint restirPathInitial = 0u;
static const uint restirPathReplay = 1u;
static const uint restirPathFinish = 2u;

static const uint restirReplayEmpty = 0u;
static const uint restirReplayDone = 1u;
//! A replay in flight at the next stage was lost or found no light; both count as a zero sample.
static const uint restirReplayInFlight = 2u;
static const uint restirReplayHasSurface = 4u;

//! The initial sampling state of a path between its wavefront stages; the path pool keeps both throughputs.
struct RestirPathRecord {
    //! Running weight sum, BSDF pdf before the reconnection vertex, Russian roulette pdf, then the packed lengths, flags and random index.
    float4 sums;
    //! Position and octahedral normal of the reconnection vertex of the path.
    float4 mainVertex;
    float4 selectedVertex;
    //! Wi pdf and partial Jacobian of the reconnection vertex of the path, then of the selected sample.
    float4 pdfs;
    float4 radiance;
    float4 targetFunction;
};

//! One random replay: the vertex before the reconnection vertex, throughput or radiance, BSDF value (half4) and pdf toward the target, light set in bits 24+.
struct RestirReplayRecord {
    float3 position;
    uint normal;
    float4 radiance;
    uint2 bsdf;
    uint material;
    float bsdfPdf;
};

//! A volume scatter of an initial path that the wavefront path tracer continues, with the pool fields it writes back.
struct RestirForkRecord {
    float4 origin;
    float4 direction;
    float4 throughput;
    uint4 state;
    float t;
    uint volume;
    uint count;
    uint frame;
    uint lightPath;
    uint unused0;
    uint unused1;
    uint unused2;
};

//! The modes of the ReSTIR DI kernel: the initial and temporal resampling, then the spatial resampling and the shading.
static const uint restirDirectSample = 0u;
static const uint restirDirectShade = 1u;

//! The RTXDI parameters of ReSTIR DI: the temporal pass writes the initial reservoirs over temporalResamplingOutputBufferIndex.
struct RestirDirectParameters {
    RTXDI_ReservoirBufferParameters reservoirBuffer;
    RTXDI_DIBufferIndices bufferIndices;
    RTXDI_DITemporalResamplingParameters temporalResampling;
    RTXDI_DISpatialResamplingParameters spatialResampling;
    //! 1 resamples the reservoirs of the previous frame.
    uint temporal;
    //! The light candidates of the initial sampling of a pixel.
    uint candidates;
    uint unused0;
    uint unused1;
};

struct RestirFrame {
    Ptr<WavefrontFrame> wavefront;
    Ptr<RTXDI_PackedPTReservoir> reservoirs;
    Ptr<float2> neighborOffsets;
    //! Two halves of width * height surfaces and closures; the half of this frame is current, the other one is the previous frame.
    Ptr<RestirSurfaceRecord> surfaces;
    Ptr<PackedClosure> closures;
    //! One record per path pool slot.
    Ptr<RestirPathRecord> paths;
    Ptr<RestirReplayRecord> replays;
    //! The state of each replay, see restirReplay*.
    Ptr<uint> replayStates;
    //! Three lists of the pixels whose resampling waits for replays: stage s writes list s % 3 and reads list (s + 2) % 3.
    Ptr<uint> pending;
    //! The length of each pending list.
    AtomicPtr<uint> counters;
    uint pendingCapacity;
    uint width;
    uint height;
    uint current;
    uint history;
    uint unused;
    //! One fork per pixel, or null when the scene has no volumes.
    Ptr<RestirForkRecord> forks;
    RTXDI_RuntimeParameters runtime;
    RTXDI_PTParameters pt;
    RadianceCacheRecord cache;
    //! ReSTIR DI: the reservoirs and the primary vertices of the wavefront paths; null in ReSTIR PT.
    Ptr<RTXDI_PackedDIReservoir> lightReservoirs;
    Ptr<DirectVertex> vertices;
    RestirDirectParameters direct;
};

//! stage and mode select the resampling pass (mode is the path mode in the path kernels), start and count the pixels of a sweep, forward its shifts per pixel.
struct RestirArguments {
    Ptr<RestirFrame> frame;
    uint parity;
    uint stage;
    uint mode;
    uint start;
    uint count;
    uint forward;
};

NR_NAMESPACE_END

#if !defined(__SLANG__)
#include <cstddef>
static_assert(sizeof(RTXDI_PackedPTReservoir) == 80);
static_assert(sizeof(RTXDI_PTParameters) == 240);
static_assert(sizeof(NoorRay::Shaders::RestirPathRecord) == 96);
static_assert(sizeof(NoorRay::Shaders::RestirReplayRecord) == 48);
static_assert(sizeof(NoorRay::Shaders::RestirForkRecord) == 96);
static_assert(offsetof(NoorRay::Shaders::RestirFrame, forks) == 104);
static_assert(offsetof(NoorRay::Shaders::RestirFrame, runtime) == 112);
static_assert(offsetof(NoorRay::Shaders::RestirFrame, pt) == 128);
static_assert(offsetof(NoorRay::Shaders::RestirFrame, cache) == 368);
static_assert(sizeof(RTXDI_PackedDIReservoir) == 24);
static_assert(sizeof(NoorRay::Shaders::RestirDirectParameters) == 144);
static_assert(offsetof(NoorRay::Shaders::RestirFrame, lightReservoirs) == 464);
static_assert(offsetof(NoorRay::Shaders::RestirFrame, direct) == 480);
static_assert(sizeof(NoorRay::Shaders::RestirFrame) == 624);
static_assert(sizeof(NoorRay::Shaders::RestirArguments) == 32);
static_assert(offsetof(RTXDI_PTReconnectionParameters, pad1) == offsetof(RTXDI_PTReconnectionParameters, reconnectionMode) + 4);
#endif
