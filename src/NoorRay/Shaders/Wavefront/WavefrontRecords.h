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

//! The indirect dispatches of the grid stages, which the advance phases size to the work counts; an empty stage launches no group.
static const uint gridActive = 0u;
static const uint gridBinSource = 1u;
static const uint gridSurface = 2u;
static const uint gridMedium = 3u;
static const uint gridShadow = 4u;
static const uint gridFree = 5u;
//! All binned paths, surfaces and media.
static const uint gridBinned = 6u;
static const uint gridSlotCount = 7u;

static const uint advanceBins = 0u;
static const uint advanceShadow = 1u;
static const uint advanceRegenerate = 2u;
static const uint advanceSwap = 3u;
static const uint advanceReset = 4u;
static const uint advanceGenerate = 5u;
//! The bins phase reads its source from bit 8: the active queue, or the alpha candidates of shadow rays.
static const uint advanceSourceShadow = 1u << 8;
//! Bit 9 runs the swap phase after the phase, which saves a dispatch when no stage runs between them.
static const uint advanceThenSwap = 1u << 9;

//! Bin 0 holds the Gaussian and volume hits; the surface hits follow in one bin per material program: the materials without a graph, then each graph.
static const uint mediumBin = 0u;
static const uint fixedMaterialBin = 1u;
static const uint firstGraphBin = 2u;
//! The material kernels read their source from bit 31: the shade queue of the paths, or of the alpha candidates of shadow rays.
static const uint materialSourceShadow = 1u << 31;
//! With this bit, the kernel of a graph bakes the emission of the triangle lights of its program instead.
static const uint materialSourceEmitters = 1u << 30;
//! The threads of one dispatch row of the emitter bake.
static const uint emitterRowThreads = 4096u * 64u;

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
//! Bits 16 to 19 of the path flags count its volume scattering events.
static const uint pathVolumeShift = 16u;
static const uint pathVolumeMask = 0xfu << pathVolumeShift;
//! Bits 20 to 25 hold the light set of the last scattering vertex, whose lights it samples and whose emitter hits count.
static const uint pathLightSetShift = 20u;
static const uint pathLightSetMask = 0x3fu << pathLightSetShift;
//! A path that ReSTIR PT forked at a volume scatter: it adds light to its pixel but is not a sample of it.
static const uint pathFlagFork = 1u << 26;
//! A fork of the camera segment: its light is in front of the primary vertex, so it goes to the fog light of the pixel.
static const uint pathFlagCameraFork = 1u << 27;
//! A path of a cache update pixel, which feeds the SHaRC radiance cache at its vertices.
static const uint pathFlagCacheUpdate = 1u << 28;
//! The modes of the radiance cache kernel.
static const uint radianceCacheModeResolve = 0u;
static const uint radianceCacheModeClear = 1u;
//! The primary AOVs of the path are those of the medium in front of its first surface, which later vertices keep.
static const uint pathFlagVolumeGuide = 1u << 29;
//! The ray after the last vertex or the volume bounce limit, as Cycles' terminate on next surface: it adds the light it finds and ends there.
static const uint pathFlagFinalRay = 1u << 30;
static const uint pathDepthMask = 0xffu;
//! A packed octahedral direction that packOctahedral never makes: the path has no known primary direction.
static const uint noPrimaryDirection = 0x80008000u;

//! The lighting split of a pixel: specular and emission rgb, hit distance sums and weights, fog, diffuse and specular luminance times light direction.
static const uint lightingSpecular = 0u;
static const uint lightingEmission = 3u;
static const uint lightingDiffuseHit = 6u;
static const uint lightingSpecularHit = 8u;
static const uint lightingFog = 10u;
static const uint lightingDiffuseSh = 13u;
static const uint lightingSpecularSh = 16u;
static const uint lightingStride = 19u;

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
    //! The material of the hit, written by the material kernel of its bin.
    Ptr<PackedClosure> closures;
    //! The last scattering event and the Cycles bounce counts of the path, which the Light Path node reads; Wavefront/PathState.slang packs it.
    Ptr<uint2> lightPath;
    //! The smallest pdf of the scatter directions of the path, as Cycles' min_ray_pdf, from which Filter Glossy blurs.
    Ptr<float> minRayPdf;
    //! The octahedral direction in which the path left its primary vertex, or noPrimaryDirection; RELAX takes it for the light direction.
    Ptr<uint> primaryDirection;
    //! The packed light receiver of the vertex that the ray left (Lights/SgLighting.slang); the light tree pdf of MIS needs it.
    Ptr<uint4> receiver;
    uint capacity;
    //! Aligns shadowEntry and keeps the record at 144 bytes, so the uint4 fields after it in the frame stay 16-byte aligned.
    uint padding;
    //! The visibility cache entry of the shadow ray, cell times visibilityChoices plus choice, or visibilityNoEntry.
    Ptr<uint> shadowEntry;
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
    //! The thread groups of the material kernel of each bin, three uints per bin, written with binCursors.
    Ptr<uint> binDispatch;
    uint binCount;
    uint sampleCapacity;
    //! The threads of the fixed grid that every stage dispatches; each thread strides over the stage's count by this.
    uint gridThreads;
    uint unused0;
    //! The thread groups of each grid stage, three uints per slot.
    Ptr<uint> gridDispatch;
};

//! The film of the wavefront path tracer. output is the displayed linear RGB.
struct FilmRecord {
    //! rgb sum and sample count per pixel, then per pixel the samples that saw nothing, filled with atomics while paths end.
    AtomicPtr<float> frame;
    //! The accumulated rgb sum and sample count.
    Ptr<float4> accumulated;
    Ptr<float4> output;
    Ptr<float4> albedo;
    Ptr<float4> normalDepth;
    Ptr<uint> objectIds;
    //! Unused, credit, Sobol index base, the accumulated samples that saw no surface.
    Ptr<uint4> pixels;
    //! The lighting split for the denoiser, lightingStride atomic floats per pixel; used only when splitsLighting is 1.
    AtomicPtr<float> lighting;
    //! Per pixel the diffuse albedo and roughness, then the specular albedo, of the primary hit; written with lighting.
    Ptr<float4> demodulation;
    uint width;
    uint height;
    uint accumulate;
    uint splitsLighting;
    //! A storage image of the film size that another API shares; the passes that write output also write it when it is set.
    StorageTextureHandle shared;
    //! 1 when the specular demodulation albedo is the EnvBRDFApprox2 albedo that DLSS-RR takes.
    uint approximatesSpecular;
    //! The factor of the output.
    float exposure;
    uint3 padding;
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

//! The shadow ray has no cell or no learned choice.
static const uint visibilityNoEntry = 0xffffffffu;

//! The learned visibility of the first light choice per cell of a hash grid over position and normal, after Pantaleoni 2019.
struct VisibilityCacheRecord {
    //! A nonzero key per used cell; Shade inserts keys, and the resolve pass evicts the cells that saw no shadow rays for long.
    AtomicPtr<uint> keys;
    //! Per cell and choice the shadow rays of this frame that arrived and that were traced.
    AtomicPtr<uint> counts;
    //! visibilityCellWords per cell: per choice the decayed counts of earlier frames as half2, then the cut; fixed during a frame so that NEE and MIS see the same pdf.
    Ptr<uint> cells;
    //! The cell count minus one; the cell count is a power of two.
    uint mask;
    uint padding;
};

//! The primary surface of a pixel: depth 0 means none, normal is the octahedral geometric normal toward the viewer, material bit 31 is backfacing.
struct RestirSurfaceRecord {
    float3 position;
    float depth;
    uint normal;
    uint view;
    uint material;
    uint instance;
};

static const uint directVertexLast = 1u;
static const uint directVertexSecondaryTerminated = 2u;
static const uint directVertexRgb = 4u;
static const uint directVertexDispersion = 8u;

//! The primary vertex of a path whose direct light ReSTIR DI resamples: Shade writes it instead of the shadow ray of the vertex.
struct DirectVertex {
    //! The throughput of the light sample of the vertex, zero where the path lights a subsurface exit.
    float4 throughput;
    //! The packed light receiver of the vertex, which the MIS weight against the BSDF hits of the path takes.
    uint4 receiver;
    uint hero;
    uint flags;
    //! The frame that wrote the vertex; a pixel whose path found no surface this frame keeps an older one.
    uint frame;
    uint unused;
};

//! The SHaRC state of an update path: the cache entries of its last two vertices and their RGB weights; length ~0 stops the update.
struct RadianceCacheState {
    uint2 indices;
    float3 weight0;
    float3 weight1;
    uint length;
    //! The wavefront path tracer defers the scatter weight to the next vertex and keeps the NEE of the vertex in direct until Shadow resolves it.
    float3 pending;
    float3 direct;
    uint unused;
};

//! The SHaRC world-space radiance cache; hashes is null when it is off. A path ends in the cache at the first vertex of depth >= depth that has an entry.
struct RadianceCacheRecord {
    //! The 64-bit keys as uint2.
    Ptr<uint2> hashes;
    //! SharcAccumulationData.
    Ptr<uint4> accumulation;
    //! SharcPackedData.
    Ptr<uint4> resolved;
    //! One state per path pool slot.
    Ptr<RadianceCacheState> states;
    float3 camera;
    float sceneScale;
    float3 previousCamera;
    uint capacity;
    uint depth;
    //! One pixel in each updateStride x updateStride block updates the cache.
    uint updateStride;
    float radianceScale;
    uint accumulationFrames;
    uint frameIndex;
    uint unused0;
    uint unused1;
    uint unused2;
};

//! The G-buffer that Shade writes for ReSTIR DI; surfaces is null when ReSTIR DI is off.
struct DirectLightingRecord {
    //! Two halves of width * height surfaces and closures; base is the first one of this frame.
    Ptr<RestirSurfaceRecord> surfaces;
    Ptr<PackedClosure> closures;
    Ptr<DirectVertex> vertices;
    uint base;
    uint frame;
};

//! The per-frame constants of every wavefront kernel.
struct WavefrontFrame {
    Ptr<SceneRecord> scene;
    Ptr<CameraRecord> camera;
    PathPoolRecord pool;
    QueueRecord queues;
    FilmRecord film;
    //! Max depth with the volume bounces from bit 8, Russian roulette start depth, frame seed, Sobol scramble seed of the accumulation.
    uint4 settings;
    //! Indirect clamp (0 off), path budget, then the fixed sub-pixel position of every primary ray in [0, 1)^2, or negative for a random one.
    float4 limits;
    //! Gaussian mode, ReSTIR on (then ReSTIR shading writes the film, not the camera paths), alpha shadow layers, regeneration on.
    uint4 options;
    RasterRecord raster;
    //! x: the pixel filter width in pixels (Cycles' filter width, 0 the pixel box); y: the direct clamp; z: 1 over Cycles' Filter Glossy; 0 is off.
    float4 filter;
    //! The diffuse, glossy and transmission bounces, as Cycles' per-kind limits; w is 1 when the last vertex traces a final ray, as Cycles.
    uint4 bounces;
    VisibilityCacheRecord visibility;
    DirectLightingRecord direct;
    RadianceCacheRecord cache;
    //! x: the vertices below this depth pick their lights by the light tree, the deeper ones by power alone.
    uint4 lighting;
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
static_assert(sizeof(NoorRay::Shaders::PathPoolRecord) == 144);
static_assert(sizeof(NoorRay::Shaders::QueueRecord) == 112);
static_assert(sizeof(NoorRay::Shaders::FilmRecord) == 112);
static_assert(sizeof(NoorRay::Shaders::RasterRecord) == 32);
static_assert(offsetof(NoorRay::Shaders::RasterRecord, draws) == 16);
static_assert(sizeof(NoorRay::Shaders::WavefrontArguments) == 16);
static_assert(sizeof(NoorRay::Shaders::VisibilityCacheRecord) == 32);
static_assert(sizeof(NoorRay::Shaders::RestirSurfaceRecord) == 32);
static_assert(sizeof(NoorRay::Shaders::DirectVertex) == 48);
static_assert(sizeof(NoorRay::Shaders::DirectLightingRecord) == 32);
static_assert(offsetof(NoorRay::Shaders::WavefrontFrame, direct) % 16 == 0);
static_assert(sizeof(NoorRay::Shaders::RadianceCacheState) == 64);
static_assert(sizeof(NoorRay::Shaders::RadianceCacheRecord) == 96);
static_assert(offsetof(NoorRay::Shaders::WavefrontFrame, cache) % 16 == 0);
#endif
