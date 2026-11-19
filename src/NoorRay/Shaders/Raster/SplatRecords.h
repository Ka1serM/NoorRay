#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/Shared.h"
#if !defined(__SLANG__)
#include "NoorRay/Shaders/Common/CameraRecords.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"
#endif

NR_NAMESPACE_BEGIN

static const uint splatGroupSize = 128u;
//! Blocks per dispatch row: a larger sort folds into rows, below the limit of 65535 workgroups per dimension.
static const uint splatRowGroups = 32768u;
static const uint splatDigits = 16u;
static const uint splatDigitBits = 4u;
//! Log depth quantized to 16 bits keeps the relative depth error near 1e-4 over a 1e5 depth range, in four sort passes.
static const uint splatKeyBits = 16u;
//! Histogram entries per workgroup of the hierarchical scan, eight per thread.
static const uint splatScanTile = 1024u;

static const uint splatCounterVisible = 0u;
static const uint splatCounterMinDepth = 1u;
static const uint splatCounterMaxDepth = 2u;
static const uint splatCounterCount = 3u;

static const uint splatStateVisible = 0u;
static const uint splatStateBlocks = 1u;
static const uint splatStatePartials = 2u;
static const uint splatStateLogMin = 3u;
static const uint splatStateScale = 4u;
static const uint splatStateCount = 5u;

static const uint splatDispatchBlocks = 0u;
static const uint splatDispatchScan = 1u;
static const uint splatDispatchCount = 2u;

//! The root arguments of every splat kernel and draw; one sort pass reads keys and values and writes the sorted pair.
struct SplatArguments {
    Ptr<SceneRecord> scene;
    Ptr<CameraRecord> camera;
    //! The depth of each visible splat as float bits until the first pass quantizes it.
    Ptr<uint> keys;
    //! The Gaussian of each visible splat.
    Ptr<uint> values;
    Ptr<uint> sortedKeys;
    Ptr<uint> sortedValues;
    //! Digit-major counts of every block, scanned in place to the first target of each digit and block.
    Ptr<uint> histogram;
    Ptr<uint> partials;
    //! Two per Gaussian: unjittered centre in NDC, NDC depth and opacity; then the half pixel axes and the half linear RGB.
    Ptr<uint4> splats;
    Ptr<uint> state;
    Ptr<DispatchSize> dispatches;
    Ptr<DrawIndexedArgs> draw;
    AtomicPtr<uint> counters;
    float2 jitter;
    uint count;
    uint shift;
    uint cloudCount;
    uint pad;
};

NR_NAMESPACE_END

#if !defined(__SLANG__)
#include <cstddef>
static_assert(sizeof(NoorRay::Shaders::SplatArguments) == 128);
static_assert(offsetof(NoorRay::Shaders::SplatArguments, jitter) == 104);
#endif
