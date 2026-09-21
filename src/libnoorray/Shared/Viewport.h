#pragma once

#include "Math.h"
#include "Types.h"

#ifdef __cplusplus
#include <cstddef>
namespace nr::graphics {
#endif

// Colour of the selection outline, and of a selected light's icon.
static const float3 ViewportSelectionColor = float3(1.0, 1.0, 0.0);

struct ViewportBillboard
{
    float4 positionType;
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
    // Icon half-size in pixels: maxRadius at nearDistance or closer, easing
    // down to minRadius at farDistance and beyond (view-space distance).
    float minRadius;
    float maxRadius;
    float nearDistance;
    float farDistance;
    // Multiplies the icon radius: 1 for the drawn icons, less for the pick
    // circles stamped into the light-id buffer.
    float radiusScale;
    uint billboardCount;
    // One uint per output pixel, row stride lightIdStride: 0 where no light
    // icon covers the pixel, otherwise the billboard index + 1.
    GpuPtr(uint) lightIds;
    uint lightIdStride;
    // Billboard index of the selected light, drawn in the selection outline's
    // colour; ~0u when no light is selected.
    uint selectedBillboard;
};

struct ViewportBillboardRoot
{
    GpuPtr(ViewportBillboardPushConstants) arguments;
};

#ifdef __cplusplus
static_assert(offsetof(ViewportCompositePushConstants, positionImage) == 20);
static_assert(offsetof(ViewportCompositePushConstants, overdraw) == 24);
static_assert(offsetof(ViewportCompositePushConstants, width) == 52);
static_assert(offsetof(ViewportCompositePushConstants, selectionSdfImage) == 64);
static_assert(sizeof(ViewportCompositePushConstants) == 72);
static_assert(offsetof(ViewportBillboardPushConstants, targetScale) == 80);
static_assert(offsetof(ViewportBillboardPushConstants, radiusScale) == 104);
static_assert(offsetof(ViewportBillboardPushConstants, lightIds) == 112);
static_assert(sizeof(ViewportBillboardPushConstants) == 128);
} // namespace nr::graphics
#endif
