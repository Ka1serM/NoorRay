#pragma once

#include "Types.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

struct Environment
{
    float3 color;
    float rotationSin;
    float rotationCos;
    float visibleExposureScale;
    float lightingExposureScale;
    float importanceWeight;
    // Resource-descriptor-heap indices; 0 means no texture.
    uint texture;
    uint cdfTexture;
    // Unreal's sky light controls, as LightControls holds them for lights.
    uint castShadows;
    float indirectLightingIntensity;
    int cdfWidth;
    int cdfHeight;
    int mapping;
    int padding0;
    float4 environmentFromWorld[3];
    // Below the environment's horizon the texture's value blends toward rgb
    // by a, as Unreal's sky light does with Lower Hemisphere Is Solid Color.
    float4 lowerHemisphere;
};

#ifdef __cplusplus
inline constexpr std::uint32_t EnvironmentNoTexture = 0;
} // namespace nr::graphics
#endif
