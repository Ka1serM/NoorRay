#pragma once

#include "Raytracing/GaussianProxyBlas.h"

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
    int maxBounces{10};
    int russianRouletteStartBounce{3};
    float exposure{};
    bool tonemappingEnabled{false};
    bool transparentBackground{false};
    float gaussianCutoffSigma{3.0f};
    GaussianProxyType gaussianProxyType{GaussianProxyType::Icosahedron};
    GaussianShadingMode gaussianShadingMode{GaussianShadingMode::GlobalIllumination};
    bool gaussianProxyOverdrawVisualization{false};
    int gaussianProxyOverdrawMax{1024};
    BufferVisualization bufferVisualization{BufferVisualization::Beauty};
};
