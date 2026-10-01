#pragma once

#include <array>
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
    // Mip bias for textures sampled at the render resolution, FidelityFX's
    // log2(render / output) - 1: the output resolves texels finer than one
    // render pixel. Zero when off.
    float textureLodBias(Extent render, Extent output) const;
    // The image the composite writes: the color target FSR reads, or the
    // output image itself when off.
    std::uint32_t compositeTarget(const RenderTargets& targets,
        noorrhi::TextureHandle output) const;

    // Recreates the FSR context for render and output rectangles up to these
    // sizes, or releases it while the mode is off. The GPU must be idle.
    // Inside them both rectangles may change every frame and FSR carries its
    // history across.
    void resize(Extent maxRender, Extent maxOutput);
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
};
