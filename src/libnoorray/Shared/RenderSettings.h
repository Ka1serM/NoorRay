#pragma once

// Render configuration shared by scene authoring and shader-facing code.
// Keep this record free of engine or graphics API dependencies so it can be
// included from both C++ and Slang.

#ifdef __cplusplus
#include <cstdint>
#endif

// Which renderer draws the scene. Spectral is the reference path tracer;
// Realtime is a biased RGB path tracer for interactive preview.
enum class RaytracerType : int
{
    Spectral,
    Realtime,
};

enum class SphericalHarmonicsOrder : int
{
    Degree0 = 0,
    Degree1 = 1,
    Degree2 = 2,
    Degree3 = 3,
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

// Stages of the realtime renderer. Each has an Off mode, in which the frame
// passes through that stage unchanged.
//
// FSR's quality modes: how much smaller than the output the realtime renderer
// traces. NativeAA traces every output pixel and leaves FSR as temporal
// anti-aliasing; the others trace 1.5x, 1.7x, 2x and 3x fewer pixels per axis.
// Off traces the output resolution, unjittered, straight into the output.
enum class UpscalerMode : int
{
    Off,
    NativeAA,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
};

// How much coarser than the render resolution the realtime renderer lights:
// ThreeQuarter, Half, Third and Quarter trace, resample and denoise lighting at 3/4, 1/2, 1/3 or
// 1/4 of the render resolution per axis. Primary visibility and the material factors stay
// at the render resolution, and the composite resolves the lighting against
// them (with the denoiser's SH form, which these modes switch on).
enum class LightingResolution : int
{
    Full,
    ThreeQuarter,
    Half,
    Third,
    Quarter,
};

// NRD's denoisers. Off composites the noisy radiance as it is.
enum class DenoiserMode : int
{
    Off,
    Relax,
};

enum class GaussianShadingMode : int
{
    GlobalIllumination,
    DirectColor,
};

struct RenderSettings
{
    int samples;
    int maxSamples;
    int maxBounces;
    float indirectLightClamp;
    bool tonemappingEnabled;
    bool transparentBackground;
    float cameraExposure;
    GaussianShadingMode gaussianShadingMode;
    SphericalHarmonicsOrder gaussianRenderSphericalHarmonics;
    bool gaussianProxyOverdrawVisualization;
    int gaussianProxyOverdrawMax;
    BufferVisualization bufferVisualization;
    RaytracerType raytracer;
    // Realtime renderer stages.
    // RELAX is the quality-first choice for ReSTIR DI.  It retains the
    // direct-light hit-distance guide needed to stop hard local-light shadow
    // boundaries from bleeding during the spatial passes.
    DenoiserMode denoiserMode;
    UpscalerMode upscalerMode;
    LightingResolution lightingResolution;

#ifdef __cplusplus
    // Slang reads this record through Frame.h and has no member initializers,
    // so the defaults live in a C++-only constructor.
    RenderSettings()
        : samples(1)
        , maxSamples(3000)
        , maxBounces(0)
        , indirectLightClamp(10.0f)
        , tonemappingEnabled(false)
        , transparentBackground(true)
        , cameraExposure(0.0f)
        , gaussianShadingMode(GaussianShadingMode::DirectColor)
        , gaussianRenderSphericalHarmonics(SphericalHarmonicsOrder::Degree3)
        , gaussianProxyOverdrawVisualization(false)
        , gaussianProxyOverdrawMax(1024)
        , bufferVisualization(BufferVisualization::Beauty)
        , raytracer(RaytracerType::Realtime)
        , denoiserMode(DenoiserMode::Relax)
        , upscalerMode(UpscalerMode::Quality)
        , lightingResolution(LightingResolution::Full)
    {
    }

    bool operator==(const RenderSettings&) const = default;
#endif
};


#ifdef __cplusplus
inline constexpr float upscalerRatio(const UpscalerMode mode)
{
    switch (mode)
    {
    case UpscalerMode::Off: return 1.0f;
    case UpscalerMode::NativeAA: return 1.0f;
    case UpscalerMode::Quality: return 1.5f;
    case UpscalerMode::Balanced: return 1.7f;
    case UpscalerMode::Performance: return 2.0f;
    case UpscalerMode::UltraPerformance: return 3.0f;
    }
    return 1.0f;
}

inline constexpr float lightingScale(const LightingResolution resolution)
{
    switch (resolution)
    {
    case LightingResolution::Full: return 1.0f;
    case LightingResolution::Half: return 2.0f;
    case LightingResolution::Third: return 3.0f;
    case LightingResolution::Quarter: return 4.0f;
    case LightingResolution::ThreeQuarter: return 4.0f / 3.0f;
    }
    return 1.0f;
}

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
