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
    int gaussianProxyType{1}; // GaussianProxyType::Octahedron
    BufferVisualization bufferVisualization{BufferVisualization::Beauty};
};
