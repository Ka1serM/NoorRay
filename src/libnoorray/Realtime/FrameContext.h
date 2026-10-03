#pragma once

#include <array>
#include <cstdint>

struct Extent
{
    uint32_t width{};
    uint32_t height{};

    bool operator==(const Extent&) const = default;
};

// What every realtime stage needs to know about the frame it records. The
// realtime renderer builds it once per frame; stages only read it.
struct FrameContext
{
    // Primary visibility is traced at `render` and lighting at `lighting`
    // (render divided by lightingScale); the output images hold `output`.
    // Each is a rectangle of images that may be allocated larger.
    Extent render;
    Extent lighting;
    // The previous frame's lighting rectangle, which the temporal stages
    // reproject from; `lighting` after a history reset.
    Extent previousLighting;
    float lightingScale{1.0f};
    Extent output;
    // Unjittered, column-major, in NRD's conventions. After a history reset
    // the previous matrices equal the current ones.
    std::array<float, 16> worldToView{};
    std::array<float, 16> viewToClip{};
    std::array<float, 16> previousWorldToView{};
    std::array<float, 16> previousViewToClip{};
    // Render pixel p samples the scene at p + 0.5 + jitter; previousJitter is
    // what the previous frame traced through.
    std::array<float, 2> jitter{};
    std::array<float, 2> previousJitter{};
    float nearPlane{};
    float verticalFieldOfView{};
    // Wall time since the previous frame; a nominal 60 Hz frame for the first.
    float frameTimeMilliseconds{};
    // Scale the shaders apply to radiance; every history holds exposed values.
    float exposureScale{};
    // The first frame of a new history: every temporal stage restarts.
    bool resetHistory{};
};
