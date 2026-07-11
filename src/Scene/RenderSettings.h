#pragma once

enum class BufferVisualization : int
{
    Beauty,
    Albedo,
    Normal,
    Cryptomatte,
    Position,
};

enum class GaussianProxyType : int;

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
    int tonemappingEnabled{1};
    int transparentBackground{};
    float gaussianCutoffSigma{3.0f};
    int gaussianProxyType{0}; // GaussianProxyType::Icosahedron
    GaussianShadingMode gaussianShadingMode{GaussianShadingMode::GlobalIllumination};
    int gaussianProxyOverdrawVisualization{};
    int gaussianProxyOverdrawMax{1024};
    BufferVisualization bufferVisualization{BufferVisualization::Beauty};
};
