#pragma once

// Render configuration shared by scene authoring and shader-facing code.
// Keep this record free of engine or graphics API dependencies so it can be
// included from both C++ and Slang.

#ifdef __cplusplus
#include <cstdint>
#define NR_RENDER_SETTINGS_DEFAULT(value) {value}
#else
#define NR_RENDER_SETTINGS_DEFAULT(value)
#endif

enum class SphericalHarmonicsOrder : int
{
    Degree0 = 0,
    Degree1 = 1,
    Degree2 = 2,
    Degree3 = 3,
};

enum class GaussianProxyType : int
{
    Icosphere,
    Octahedron,
    Icosahedron,
    IcosphereLevel2,
};

enum class BufferVisualization : int
{
    Beauty,
    Albedo,
    Normal,
    Cryptomatte,
    Position,
    ProxyOverdraw,
};

enum class GaussianShadingMode : int
{
    GlobalIllumination,
    DirectColor,
};

struct RenderSettings
{
    int samples NR_RENDER_SETTINGS_DEFAULT(1);
    int maxSamples NR_RENDER_SETTINGS_DEFAULT(3000);
    bool aovEnabled NR_RENDER_SETTINGS_DEFAULT(true);
    int maxBounces NR_RENDER_SETTINGS_DEFAULT(10);
    float indirectLightClamp NR_RENDER_SETTINGS_DEFAULT(10.0f);
    bool tonemappingEnabled NR_RENDER_SETTINGS_DEFAULT(false);
    bool transparentBackground NR_RENDER_SETTINGS_DEFAULT(false);
    float cameraExposure NR_RENDER_SETTINGS_DEFAULT();
    float gaussianCutoffSigma NR_RENDER_SETTINGS_DEFAULT(3.0f);
    GaussianProxyType gaussianProxyType
        NR_RENDER_SETTINGS_DEFAULT(GaussianProxyType::IcosphereLevel2);
    GaussianShadingMode gaussianShadingMode
        NR_RENDER_SETTINGS_DEFAULT(GaussianShadingMode::DirectColor);
    SphericalHarmonicsOrder gaussianRenderSphericalHarmonics
        NR_RENDER_SETTINGS_DEFAULT(SphericalHarmonicsOrder::Degree3);
    bool gaussianProxyOverdrawVisualization
        NR_RENDER_SETTINGS_DEFAULT(false);
    int gaussianProxyOverdrawMax NR_RENDER_SETTINGS_DEFAULT(1024);
    BufferVisualization bufferVisualization
        NR_RENDER_SETTINGS_DEFAULT(BufferVisualization::Beauty);

#ifdef __cplusplus
    bool operator==(const RenderSettings&) const = default;
#endif
};

#undef NR_RENDER_SETTINGS_DEFAULT

#ifdef __cplusplus
inline bool rendersProxyOverdraw(const RenderSettings& settings)
{
    return settings.gaussianProxyOverdrawVisualization;
}

inline constexpr uint32_t sphericalHarmonicsCoefficientCount(
    const SphericalHarmonicsOrder order)
{
    const uint32_t degree = static_cast<uint32_t>(order);
    return (degree + 1) * (degree + 1);
}

inline constexpr SphericalHarmonicsOrder clampSphericalHarmonicsOrder(
    const int degree)
{
    return degree <= 0 ? SphericalHarmonicsOrder::Degree0
        : degree == 1 ? SphericalHarmonicsOrder::Degree1
        : degree == 2 ? SphericalHarmonicsOrder::Degree2
                      : SphericalHarmonicsOrder::Degree3;
}
#endif
