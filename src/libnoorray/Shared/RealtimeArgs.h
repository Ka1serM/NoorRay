#pragma once

#include "Frame.h"
#include "RealtimeLighting.h"

// Arguments of every realtime pass, shared by the host and the Slang passes.
// Each stage of the realtime renderer (Realtime/*) owns one block and fills it.

#ifdef __cplusplus
namespace nr::graphics {
#endif

// The camera of this frame and the previous one. The matrices are unjittered,
// column-major, in NRD's conventions (left-handed view, clip Y up with image
// row 0 at +1). After a history reset "previous" equals "current".
struct RealtimeView
{
    float worldToClip[16];
    float previousWorldToClip[16];
    float3 cameraPosition;
    float nearPlane;
    float3 previousCameraPosition;
    // Advances every realtime frame; seeds the per-pixel paths.
    uint frameIndex;
    float3 cameraForward;
    uint outputWidth;
    // Sample position of render pixel p: p + 0.5 + jitter.
    float2 jitter;
    uint outputHeight;
    uint padding;
};

// Render-resolution images the primary pass writes and the stages read, as
// descriptor-heap storage indices. Named by meaning, not by consumer.
struct RenderTargetHandles
{
    // rgb: radiance of one lobe, divided by its demodulation factor and packed
    // for the active denoiser (see NrdSignal.slang).
    uint diffuse;
    uint specular;
    uint normalRoughness;
    uint viewZ;
    // UV offset from this frame's unjittered position to the previous one.
    uint motion;
    // Inverted device depth, infinite far plane.
    uint depth;
    // rgb: radiance that is not denoised (emission, background), a: coverage.
    uint emission;
    uint diffuseFactor;
    uint specularFactor;
    // HDR beauty the composite writes: the upscaler's input, or the output
    // image itself when nothing upscales.
    uint color;
    uint albedo;
    uint normal;
    uint position;
    uint cryptomatte;
    uint padding0;
    uint padding1;
};

// SHaRC world-space radiance cache (external/SHARC). The buffers are untyped
// here because the record types come from the SHARC headers; Sharc.slang casts
// them. A zero capacity is how the shaders see the cache switched off.
struct RadianceCacheArgs
{
    GpuPtr(uint64_t) hashEntries;
    GpuPtr(uint) accumulation;
    GpuPtr(uint) resolved;
    float sceneScale;
    float radianceScale;
    uint capacity;
    uint accumulationFrames;
    uint staleFrames;
    // Grid the update pass spreads its paths over, one path per cell. The
    // cache is world-space, so this is deliberately not the render resolution.
    uint updateGridWidth;
    uint updateGridHeight;
    uint padding;
};

// NRD (external/NRD). The composite reads `diffuse` and `specular`: NRD's
// outputs, or the render targets themselves when the denoiser is off.
struct DenoiserArgs
{
    uint diffuse;
    uint specular;
    // Nonzero when the signals are packed for RELAX instead of REBLUR.
    uint relax;
    // View Z beyond this is background, which the primary pass writes at twice
    // the range.
    float denoisingRange;
    // REBLUR hit distance normalization (A, B, C).
    float3 hitDistanceParameters;
    uint padding;
};

struct RealtimeArgs
{
    Frame frame;
    RealtimeView view;
    RenderTargetHandles targets;
    RadianceCacheArgs radianceCache;
    DenoiserArgs denoiser;
    RealtimeLighting lighting;
};

// What every realtime launch passes: RealtimeArgs is a few kilobytes, so it is
// staged once per sample and each launch carries only its address.
struct RealtimeRoot
{
    GpuPtr(RealtimeArgs) args;
};

#ifdef __cplusplus
static_assert(sizeof(Frame) % 8 == 0);
static_assert(offsetof(RealtimeArgs, frame) == 0);
static_assert(sizeof(RealtimeView) % 16 == 0);
static_assert(sizeof(RenderTargetHandles) % 16 == 0);
static_assert(sizeof(RadianceCacheArgs) % 8 == 0);
static_assert(sizeof(DenoiserArgs) % 16 == 0);
} // namespace nr::graphics
#endif
