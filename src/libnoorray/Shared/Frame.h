#pragma once

#include "Camera.h"
#include "Environment.h"
#include "Light.h"
#include "Material.h"
#include "RenderSettings.h"
#include "Scene.h"

#ifdef __cplusplus
#include <cstddef>
namespace nr::graphics {
#endif

struct Frame
{
    Scene scene;
    GpuPtr(uint) lens;
    GpuPtr(GpuPtr(Material)) materials;
    GpuPtr(PointLight) pointLights;
    uint pointLightCount;
    GpuPtr(SpotLight) spotLights;
    uint spotLightCount;
    GpuPtr(RectLight) rectLights;
    uint rectLightCount;
    GpuPtr(DirectionalLight) directionalLights;
    uint directionalLightCount;
    GpuPtr(MeshLight) meshLights;
    uint meshLightCount;
    // Sum of every light's selection weight, for uniform-vs-weighted picking.
    float lightFiniteWeight;
    GpuPtr(uint) energyLuts;
    GpuPtr(uint) spectralTables;
    GpuPtr(Environment) environment;
    GpuPtr(float4) accumulation;
    GpuPtr(uint) gaussianRecords;
    GpuPtr(uint) gaussianOpacities;
    GpuPtr(half) gaussianShCoefficients;
    GpuPtr(uint) gaussianInstanceOffsets;
    GpuPtr(uint) gaussianOverdraw;
    uint64_t topLevelAS;
    // Resource-descriptor-heap indices of the output images. The padding
    // keeps this block at its former 40 bytes, so nothing after it moves.
    uint colorImage;
    uint albedoImage;
    uint normalImage;
    uint positionImage;
    uint cryptomatteImage;
    uint imagePadding0;
    uint imagePadding1;
    uint imagePadding2;
    uint imagePadding3;
    uint imagePadding4;

    uint width;
    uint height;
    uint frameIndex;
    uint sampleIndex;
    float shutterOpen;
    float shutterClose;
    uint gaussianCount;
    uint gaussianShCoefficientCount;
    float gaussianCutoffDistanceSq;
    uint maxBounces;
    float indirectLightClamp;
    uint transparentBackground;
    uint gaussianProxyTriangleCount;
    uint gaussianOverdrawEnabled;
    uint gaussianOverdrawMax;
    uint aovEnabled;
    uint gaussianInstanceBase;
    Camera camera;
    // Slang does not pad a struct to its alignment, C++ does: this keeps the
    // size a multiple of 8 in both, so records that follow a Frame (see
    // RealtimeArgs) sit at the same offset on the host and the GPU.
    uint padding;
};

#ifdef __cplusplus
static_assert(offsetof(Frame, colorImage) == offsetof(Frame, topLevelAS) + 8);
static_assert(offsetof(Frame, cryptomatteImage) == offsetof(Frame, colorImage) + 16);
static_assert(offsetof(Frame, width) == offsetof(Frame, colorImage) + 40);
} // namespace nr::graphics
#endif
