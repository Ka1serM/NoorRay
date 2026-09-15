#pragma once

#include "Math.h"
#include "Types.h"

#ifdef __cplusplus
#include <cstddef>
namespace nr::graphics {
#endif

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
    uint padding0;
};

struct ViewportCompositeRoot
{
    GpuPtr(ViewportCompositePushConstants) arguments;
};

struct ViewportBillboardPushConstants
{
    float4x4 viewProjection;
    GpuPtr(ViewportBillboard) billboards;
    float2 screenSize;
    float radius;
    uint padding0;
};

struct ViewportBillboardRoot
{
    GpuPtr(ViewportBillboardPushConstants) arguments;
};

#ifdef __cplusplus
static_assert(offsetof(ViewportCompositePushConstants, positionImage) == 20);
static_assert(offsetof(ViewportCompositePushConstants, overdraw) == 24);
static_assert(sizeof(ViewportCompositePushConstants) == 56);
} // namespace nr::graphics
#endif
