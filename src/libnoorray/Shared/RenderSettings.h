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

// How the realtime renderer samples light (external/RTXDI-Library). ReSTIRDI
// resamples direct light on primary surfaces with ReSTIR DI and draws light
// samples at later path vertices from RTXDI's ReGIR grid; ReSTIRGI also
// resamples the primary surfaces' diffuse indirect light with ReSTIR GI.
// SingleSample bypasses RTXDI: one power-weighted light sample per vertex.
enum class RealtimeLightingMode : int
{
    ReSTIRDI,
    ReSTIRGI,
    SingleSample,
};

// Stages of the realtime renderer. Each has an Off mode, in which the frame
// passes through that stage unchanged.
//
// SHaRC's world-space radiance cache. Off traces indirect paths to their end.
enum class RadianceCacheMode : int
{
    Off,
    Sharc,
};

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

// NRD's denoisers. Off composites the noisy radiance as it is.
enum class DenoiserMode : int
{
    Off,
    Reblur,
    Relax,
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
    RaytracerType raytracer NR_RENDER_SETTINGS_DEFAULT(RaytracerType::Realtime);
    // Realtime renderer stages.
    RadianceCacheMode radianceCacheMode NR_RENDER_SETTINGS_DEFAULT(RadianceCacheMode::Sharc);
    RealtimeLightingMode realtimeLighting
        NR_RENDER_SETTINGS_DEFAULT(RealtimeLightingMode::ReSTIRDI);
    // RELAX is the quality-first choice for ReSTIR DI.  It retains the
    // direct-light hit-distance guide needed to stop hard local-light shadow
    // boundaries from bleeding during the spatial passes.
    DenoiserMode denoiserMode NR_RENDER_SETTINGS_DEFAULT(DenoiserMode::Relax);
    UpscalerMode upscalerMode NR_RENDER_SETTINGS_DEFAULT(UpscalerMode::Quality);

#ifdef __cplusplus
    bool operator==(const RenderSettings&) const = default;
#endif
};

#undef NR_RENDER_SETTINGS_DEFAULT

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
