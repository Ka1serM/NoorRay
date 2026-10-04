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
    float3 previousPreviousCameraPosition;
    uint cameraHistoryPadding;
    float3 cameraForward;
    uint outputWidth;
    // Sample position of render pixel p: p + 0.5 + jitter, in render pixels.
    // Lighting samples move by the same distance on screen, so lighting pixel
    // p samples at p + 0.5 + jitter / lightingScale in lighting pixels:
    // jittering by whole lighting pixels would move the lighting further per
    // frame than the upscaler's history tolerates in motion.
    float2 jitter;
    uint outputHeight;
    // Lighting pixels are this many render pixels across (1 at full
    // resolution); the lighting rectangle is lightingWidth x lightingHeight.
    float lightingScale;
    uint lightingWidth;
    uint lightingHeight;
    // Row pitch of the per-lighting-pixel buffers: their allocated width,
    // which stays fixed while the rectangle inside it changes size, so this
    // and the previous frame index them alike.
    uint surfaceStride;
    // The previous frame's lighting rectangle, which its surfaces and
    // reservoirs cover.
    uint previousLightingWidth;
    uint previousLightingHeight;
    uint padding1;
    uint padding2;
};

// Images the primary passes write and the stages read, as descriptor-heap
// storage indices. Named by meaning, not by consumer.
struct RenderTargetHandles
{
    // Lighting resolution. rgb: radiance of one lobe, divided by its
    // demodulation factor and packed for the active denoiser (see
    // NrdSignal.slang).
    uint diffuse;
    uint specular;
    // Lighting resolution. xyz: each lobe's first-bounce directions, weighted
    // by luminance, for the SH denoisers (see NrdSignal.slang).
    uint diffuseSh1;
    uint specularSh1;
    // The denoiser's guides for the lighting samples, at lighting resolution.
    uint lightingNormalRoughness;
    uint lightingViewZ;
    uint lightingMotion;
    uint layerDiffuse;
    uint layerSpecular;
    uint layerLightingNormalRoughness;
    uint layerLightingViewZ;
    uint layerLightingMotion;
    // Everything below is at render resolution.
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
    uint layerDiffuseFactor;
    uint layerSpecularFactor;
    uint layerNormalRoughness;
    uint layerViewZ;
    uint transparencyMask;
    // HDR beauty the composite writes: the upscaler's input, or the output
    // image itself when nothing upscales.
    uint color;
    uint cryptomatte;
    uint padding;
    uint padding2;
};

// NRD (external/NRD). The composite reads `diffuse` and `specular`: NRD's
// outputs, or the render targets themselves when the denoiser is off. In SH
// mode those are the SH0 halves and `diffuseSh1`/`specularSh1` the SH1 ones.
struct DenoiserArgs
{
    uint diffuse;
    uint specular;
    uint diffuseSh1;
    uint specularSh1;
    uint layerDiffuse;
    uint layerSpecular;
    // Nonzero when the signals are packed for an SH denoiser. Only set while
    // a denoiser runs.
    uint sphericalHarmonics;
    // View Z beyond this is background, which the primary pass writes at twice
    // the range.
    float denoisingRange;

};

struct RealtimeArgs
{
    Frame frame;
    RealtimeView view;
    RenderTargetHandles targets;
    DenoiserArgs denoiser;
    RealtimeLighting lighting;
};

// What every realtime launch passes: RealtimeArgs is a few kilobytes, so it is
// staged once per sample and each launch carries only its address.
struct RealtimeRoot
{
    GpuPtr(RealtimeArgs) args;
};

// RealtimePick's launch: a RealtimeRoot, so realtimeArgs() reads it alike,
// followed by the render pixel whose surface position goes to `position`.
struct RealtimePickRoot
{
    GpuPtr(RealtimeArgs) args;
    GpuPtr(float4) position;
    uint pixelX;
    uint pixelY;
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
