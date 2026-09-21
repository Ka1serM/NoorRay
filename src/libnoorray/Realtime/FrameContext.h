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
    // Rays are traced at `render`; the output images hold `output`.
    Extent render;
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
    // The first frame of a new history: every temporal stage restarts.
    bool resetHistory{};
};
