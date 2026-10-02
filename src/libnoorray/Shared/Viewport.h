#pragma once

#include "Math.h"
#include "Types.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

// Colour of the selection outline, and of a selected light's icon.
static const float3 ViewportSelectionColor = float3(1.0, 1.0, 0.0);

struct ViewportBillboard
{
    float4 positionType;
    // Alpha 0 hides the icon of a hidden light.
    float4 color;
};

struct ViewportCompositePushConstants
{
    // Resource-descriptor-heap indices.
    uint colorImage;
    uint outputImage;
    uint idImage;
    uint albedoImage;
    uint normalImage;
    uint positionImage;
    GpuPtr(uint) overdraw;

    uint selectedCryptomatteId;
    float exposure;
    int bufferVisualization;
    int tonemappingEnabled;
    uint overdrawMax;
    // Logical size of the render inside the (possibly larger) images.
    uint width;
    uint height;
    // Output pixels per traced pixel (1 when the renderer traces at the
    // logical size). The AOVs the composite reads are a nearest resample of
    // the render resolution, so this is how wide one step of their silhouette
    // is on screen.
    float upscaleRatio;
    // Persistent per-pixel selection distance field: the running average of
    // every frame's estimate since it restarted.
    uint selectionSdfImage;
    // How many frames the field already averages; 0 restarts it.
    uint selectionSampleCount;
};

struct ViewportCompositeRoot
{
    GpuPtr(ViewportCompositePushConstants) arguments;
};

struct ViewportBillboardPushConstants
{
    float4x4 viewProjection;
    GpuPtr(ViewportBillboard) billboards;
    // Logical render size in pixels.
    float2 screenSize;
    // Logical size / output image size. The render target has no viewport
    // field, so the shader maps NDC into the bottom-left logical rectangle.
    float2 targetScale;
    // Fixed half-size of each camera-facing plane, in world units.
    float worldHalfSize;
    uint billboardCount;
    // One uint per output pixel, row stride lightIdStride: 0 where no light
    // icon covers the pixel, otherwise the billboard index + 1.
    GpuPtr(uint) lightIds;
    uint lightIdStride;
    // Billboard index of the selected light, drawn in the selection outline's
    // colour; ~0u when no light is selected.
    uint selectedBillboard;
    // World-space camera position. w is one when it is valid; a manually
    // supplied view-projection with no camera leaves distance fading disabled.
    float4 cameraPosition;
    // x/y: fade in from the camera; z/w: fade out at the local edit radius.
    float4 distanceFade;
};

struct ViewportBillboardRoot
{
    GpuPtr(ViewportBillboardPushConstants) arguments;
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
