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
    int diffuseBounces{3};
    int specularBounces{3};
    int transmissionBounces{3};
    int adaptiveSamplingEnabled{};
    int adaptiveMinSamples{4};
    float adaptiveTargetError{0.001f};
    int russianRouletteStartBounce{3};
    float exposure{};
    int transparentBackground{};
    BufferVisualization bufferVisualization{BufferVisualization::Beauty};
};
