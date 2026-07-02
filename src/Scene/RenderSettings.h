#pragma once

enum class BufferVisualization : int
{
    Beauty,
    Albedo,
    Normal,
    Cryptomatte,
    Position,
};

class RenderSettings
{
public:
    int samples{1};
    int maxSamples{3000};
    int diffuseBounces{64};
    int specularBounces{64};
    int transmissionBounces{64};
    int adaptiveSamplingEnabled{};
    int adaptiveMinSamples{4};
    float adaptiveTargetError{0.001f};
    int russianRouletteStartBounce{3};
    float exposure{};
    int tonemappingEnabled{1};
    int transparentBackground{};
    BufferVisualization bufferVisualization{BufferVisualization::Beauty};
};
