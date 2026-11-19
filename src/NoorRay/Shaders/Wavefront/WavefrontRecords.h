#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/Shared.h"
#if !defined(__SLANG__)
#include "NoorRay/Shaders/Common/GpuRecords.h"
#endif

NR_NAMESPACE_BEGIN

static const uint wavefrontGroupSize = 64u;

static const uint counterActive = 0u;
static const uint counterNextActive = 1u;
static const uint counterShadow = 2u;
static const uint counterNextShadow = 3u;
static const uint counterSamples = 4u;
static const uint counterSampleCursor = 5u;
static const uint counterFree = 6u;
static const uint counterCount = 7u;

static const uint advanceBins = 0u;
static const uint advanceShadow = 1u;
static const uint advanceRegenerate = 2u;
static const uint advanceSwap = 3u;
static const uint advanceReset = 4u;
static const uint advanceGenerate = 5u;
//! The bins phase reads its source from bit 8: the active queue, or the alpha candidates of shadow rays.
static const uint advanceSourceShadow = 1u << 8;

static const uint surfaceBin = 0u;
static const uint mediumBin = 1u;
static const uint shadeBinCount = 2u;

//! A primary object id is the dense instance index of a mesh, or the cloud index with this bit for a Gaussian cloud.
static const uint objectIdGaussian = 1u << 31;

static const uint hitKindMesh = 0u;
static const uint hitKindGaussian = 1u;
static const uint hitKindVolume = 2u;
static const uint hitKindMiss = 3u;

static const uint pathFlagDelta = 1u << 8;
static const uint pathFlagSecondaryTerminated = 1u << 9;
static const uint pathFlagAlphaCandidate = 1u << 10;
static const uint pathFlagPrimarySample = 1u << 11;
static const uint pathFlagShadowAlpha = 1u << 12;
static const uint pathFlagNoMis = 1u << 13;
static const uint pathFlagRgb = 1u << 14;
static const uint pathFlagRasterPrimary = 1u << 15;
static const uint pathDepthMask = 0xffu;

//! One field per buffer, indexed by path slot. shadowRadiance is spectral and becomes RGB in Shadow.
struct PathPoolRecord {
    //! w is the packed ray cone.
    Ptr<float4> origin;
    //! w is the solid angle pdf of the sampled direction, 0 after a delta event.
    Ptr<float4> direction;
    Ptr<float4> throughput;
    Ptr<float4> radiance;
    //! Pixel, Sobol sample index, depth and flags, hero u (16 bits) and the 1/rate weight as half.
    Ptr<uint4> state;
    //! t, barycentrics, primitive or Gaussian.
    Ptr<uint4> hit;
    //! kind << 30 | instance, cloud or volume.
    Ptr<uint> hitInstance;
    Ptr<float4> shadowOrigin;
    Ptr<float4> shadowDirection;
    Ptr<float4> shadowRadiance;
    Ptr<uint> pathBin;
    uint capacity;
    uint unused;
};

//! The active and shadow queues alternate; the host passes the parity.
struct QueueRecord {
    AtomicPtr<uint> counters;
    //! binCounts and binCursors: the paths of each shade bin and, after the scatter, the end of its range in shade.
    AtomicPtr<uint> binCounts;
    AtomicPtr<uint> binCursors;
    Ptr<uint> active0;
    Ptr<uint> active1;
    Ptr<uint> shadow0;
    Ptr<uint> shadow1;
    Ptr<uint> shade;
    //! Slots of ended paths for Regenerate.
    Ptr<uint> free;
    //! Pixel (bit 31 marks its first sample) and Sobol index.
    Ptr<uint2> samples;
    uint binCount;
    uint sampleCapacity;
    //! The threads of the fixed grid that every stage dispatches; each thread strides over the stage's count by this.
    uint gridThreads;
    uint unused;
};

//! The film of the wavefront path tracer. output is the displayed linear RGB.
struct FilmRecord {
    //! rgb sum and sample count per pixel, filled with atomics while paths end.
    AtomicPtr<float> frame;
    //! The accumulated rgb sum and sample count.
    Ptr<float4> accumulated;
    Ptr<float4> output;
    Ptr<float4> albedo;
    Ptr<float4> normalDepth;
    Ptr<uint> objectIds;
    //! Unused, credit, Sobol index base, unused.
    Ptr<uint4> pixels;
    uint width;
    uint height;
    uint accumulate;
    //! Keeps the record at 80 bytes, so the uint4 fields after it in the frame stay 16-byte aligned.
    uint padding0;
    uint padding1;
    uint padding2;
};

//! The primary hits of the raster pass.
struct RasterRecord {
    //! The rasterized triangle id + 1 per pixel, 0 for no triangle.
    UintTextureHandle visibility;
    uint enabled;
    //! The sub-pixel position of the rasterized sample.
    float2 jitter;
    //! First triangle id and instance of each draw, sorted by first triangle.
    Ptr<uint2> draws;
    uint drawCount;
    //! The paths shade linear sRGB instead of spectral samples.
    uint rgb;
};

//! The per-frame constants of every wavefront kernel.
struct WavefrontFrame {
    Ptr<SceneRecord> scene;
    Ptr<CameraRecord> camera;
    PathPoolRecord pool;
    QueueRecord queues;
    FilmRecord film;
    //! Max depth, Russian roulette start depth, frame seed, frame index.
    uint4 settings;
    //! Indirect clamp (0 off), path budget, unused, unused.
    float4 limits;
    //! Gaussian mode, special bin, alpha shadow layers, regeneration on.
    uint4 options;
    RasterRecord raster;
};

//! The root arguments of every wavefront kernel. parity selects the queues of this iteration, bin the shade bin or the advance phase.
struct WavefrontArguments {
    Ptr<WavefrontFrame> frame;
    uint parity;
    uint bin;
};

NR_NAMESPACE_END

#if !defined(__SLANG__)
#include <cstddef>
static_assert(sizeof(NoorRay::Shaders::PathPoolRecord) == 96);
static_assert(sizeof(NoorRay::Shaders::QueueRecord) == 96);
static_assert(sizeof(NoorRay::Shaders::FilmRecord) == 80);
static_assert(sizeof(NoorRay::Shaders::RasterRecord) == 32);
static_assert(offsetof(NoorRay::Shaders::RasterRecord, draws) == 16);
static_assert(sizeof(NoorRay::Shaders::WavefrontArguments) == 16);
#endif
