#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>

#include <noorrhi/noorrhi.hpp>

#include "Realtime/FrameContext.h"
#include "Realtime/RenderTargets.h"
#include "Shared/RenderSettings.h"

// AMD FidelityFX Super Resolution 3.1 temporal upscaling
// (external/FidelityFX-SDK, FSR 3.1.4). The mode sets how much smaller than
// the output rays are traced; each frame's rays go through a sub-pixel jitter
// FSR accumulates, and FSR writes the output image from the color, depth and
// motion targets. Off, rays are traced at the output resolution, unjittered,
// and the composite writes the output image directly.
//
// FSR runs through the SDK's own Vulkan backend on NoorRHI's device and
// records into NoorRHI's command stream.
class Upscaler
{
public:
    explicit Upscaler(noorrhi::Device& device);
    ~Upscaler();

    Upscaler(const Upscaler&) = delete;
    Upscaler& operator=(const Upscaler&) = delete;

    void setMode(UpscalerMode mode) { mode_ = mode; }
    UpscalerMode mode() const { return mode_; }

    // The resolution rays are traced at for this output.
    Extent renderExtent(Extent output) const;
    // Advances the jitter sequence, restarting it on `reset`, and
    // returns the offset in the renderer's convention: render pixel p samples
    // the scene at p + 0.5 + jitter. Zero when off.
    std::array<float, 2> nextJitter(Extent render, Extent output, bool reset);
    // The image the composite writes: the color target FSR reads, or the
    // output image itself when off.
    std::uint32_t compositeTarget(const RenderTargets& targets,
        noorrhi::TextureHandle output) const;

    // Recreates the FSR context for render and output images up to this
    // size. The GPU must be idle. A repeat of the current size is a no-op, so
    // the accumulated history survives a render-resolution change inside it.
    void resize(Extent maxOutput);
    // Upscales the color target into `output`, an RGBA32F image allocated at
    // `outputAllocation` whose frame.output rectangle holds the frame.
    void record(const FrameContext& frame, const RenderTargets& targets,
        noorrhi::ImageHandle output, Extent outputAllocation);

private:
    struct State;

    noorrhi::Device& device_;
    UpscalerMode mode_{RenderSettings{}.upscalerMode};
    std::unique_ptr<State> state_;
    std::uint32_t jitterIndex_{};
    std::chrono::steady_clock::time_point lastFrameTime_{};
};
