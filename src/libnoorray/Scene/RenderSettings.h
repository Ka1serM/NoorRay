#pragma once

#include "Backend/OptiX/Acceleration/GaussianProxyBlas.h"
#include "Materials/Shading/SphericalHarmonics.h"

enum class BufferVisualization : int
{
    Beauty,
    Albedo,
    Normal,
    Cryptomatte,
    Position,
    // Full-frame diagnostic outputs. Denoised is a separate shared AOV buffer;
    // proxy overdraw remains a beauty-surface diagnostic.
    Denoised,
    ProxyOverdraw,
};

enum class GaussianShadingMode : int
{
    GlobalIllumination,
    DirectColor,
};

class RenderSettings
{
public:
    int samples{1};
    int maxSamples{3000};
    bool aovEnabled{true};
    bool optixDenoiserEnabled{false};
    int optixDenoiserMinSamples{1};
    int maxBounces{10};
    // Blender/Cycles-style maximum per-contribution indirect light. Zero
    // disables clamping.
    float indirectLightClamp{10.0f};
    bool tonemappingEnabled{false};
    bool transparentBackground{false};
    // Applied to the rendered radiance by integrations that own the camera
    // display path (Hydra). The standalone viewport applies its camera
    // exposure during presentation instead.
    float cameraExposure{};
    float gaussianCutoffSigma{3.0f};
    // The tighter level-2 proxy significantly reduces false-positive OptiX
    // any-hit invocations in dense splat scenes while preserving coverage of
    // the exact Gaussian cutoff volume.
    GaussianProxyType gaussianProxyType{GaussianProxyType::IcosphereLevel2};
    GaussianShadingMode gaussianShadingMode{GaussianShadingMode::DirectColor};
    SphericalHarmonicsOrder gaussianRenderSphericalHarmonics{SphericalHarmonicsOrder::Degree3};
    bool gaussianProxyOverdrawVisualization{false};
    int gaussianProxyOverdrawMax{1024};
    BufferVisualization bufferVisualization{BufferVisualization::Beauty};
};

// The checkbox is the sole switch that runs the OptiX denoiser. Selecting the
// denoised buffer is only a presentation choice and must not enable work by
// itself. Proxy overdraw has no beauty sample to denoise.
inline bool runsOptixDenoiser(const RenderSettings& settings)
{
    if (settings.bufferVisualization == BufferVisualization::ProxyOverdraw)
        return false;
    return settings.optixDenoiserEnabled;
}

inline bool rendersProxyOverdraw(const RenderSettings& settings)
{
    return settings.gaussianProxyOverdrawVisualization;
}
