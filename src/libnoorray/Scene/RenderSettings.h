#pragma once

#include "Raytracing/Acceleration/GaussianProxyBlas.h"
#include "Shading/SphericalHarmonics.h"

enum class BufferVisualization : int
{
    Beauty,
    Albedo,
    Normal,
    Cryptomatte,
    Position,
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
    bool noiseLimitEnabled{false};
    float noiseLevel{0.0001f};
    bool aovEnabled{true};
    bool optixDenoiserEnabled{false};
    int optixDenoiserMinSamples{1};
    int maxBounces{10};
    int russianRouletteStartBounce{3};
    bool tonemappingEnabled{false};
    bool transparentBackground{false};
    float gaussianCutoffSigma{3.0f};
    GaussianProxyType gaussianProxyType{GaussianProxyType::Icosphere};
    GaussianShadingMode gaussianShadingMode{GaussianShadingMode::DirectColor};
    SphericalHarmonicsOrder gaussianRenderSphericalHarmonics{SphericalHarmonicsOrder::Degree3};
    bool gaussianProxyOverdrawVisualization{false};
    int gaussianProxyOverdrawMax{1024};
    BufferVisualization bufferVisualization{BufferVisualization::Beauty};
};
