#pragma once

#include "Camera.h"
#include "Environment.h"
#include "Light.h"
#include "Material.h"
#include "RenderSettings.h"
#include "Scene.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

// The beauty average's restart state, kept on the GPU in two slots picked by
// Frame::accumulationSlot. A frame's hit shaders mark its slot and its
// accumulate pass resets the other one for the next frame, so no pass reads a
// slot that another of its threads resets.
struct BeautyAccumulation
{
    // Set when a ray hits an animated material or instance.
    uint animatedHit[2];
    // Samples in the average before this frame's.
    uint sampleCount[2];
};

struct Frame
{
    Scene scene;
    GpuPtr(uint) lens;
    GpuPtr(GpuPtr(Material)) materials;
    GpuPtr(PointLight) pointLights;
    uint pointLightCount;
    uint pointLightPadding;
    GpuPtr(SpotLight) spotLights;
    uint spotLightCount;
    uint spotLightPadding;
    GpuPtr(RectLight) rectLights;
    uint rectLightCount;
    uint rectLightPadding;
    GpuPtr(DirectionalLight) directionalLights;
    uint directionalLightCount;
    uint directionalLightPadding;
    GpuPtr(MeshLight) meshLights;
    uint meshLightCount;
    // Sum of every light's selection weight, for uniform-vs-weighted picking.
    float lightFiniteWeight;
    GpuPtr(uint) energyLuts;
    GpuPtr(uint) spectralTables;
    GpuPtr(Environment) environment;
    GpuPtr(float4) accumulation;
    GpuPtr(uint) gaussianOverdraw;
    GpuPtr(BeautyAccumulation) beautyAccumulation;
    uint64_t topLevelAS;
    // Resource-descriptor-heap indices of the output images.
    uint colorImage;
    uint albedoImage;
    uint normalImage;
    uint positionImage;
    uint cryptomatteImage;
    uint materialSampler;
    // Added to the mip level of material textures sampled by ray differentials.
    float textureLodBias;
    // The realtime camera's near plane and orthographic far plane, which the
    // projection MaterialX materials read is built with.
    float nearPlane;
    float orthographicFarPlane;

    uint width;
    uint height;
    uint frameIndex;
    float gameTime;
    uint sampleIndex;
    uint accumulationSlot;
    float shutterOpen;
    float shutterClose;
    uint gaussianShCoefficientCount;
    uint maxBounces;
    float indirectLightClamp;
    uint transparentBackground;
    uint gaussianOverdrawEnabled;
    uint gaussianOverdrawMax;
    Camera camera;
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
