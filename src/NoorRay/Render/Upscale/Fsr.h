#pragma once

#include <cstdint>
#include <expected>
#include <memory>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Render/PipelineCache.h"

namespace NoorRay::Render {

//! The camera of one FSR frame; the inputs are the images of Fsr.
struct UpscaleFrame {
    //! The sub-pixel sample position of the primary rays in [0, 1).
    NoorRHI::float2 jitter{0.5f, 0.5f};
    float near = 0.1f;
    float verticalFov = 1.0f;
    float frameTime = 16.6f;
    float sharpness = 0.2f;
    bool reset = false;
};

//! AMD FSR 3.1 upscaling on NoorRHI: the FidelityFX library plans the passes of the unmodified FSR shaders through a NoorRHI backend.
class Fsr {
public:
    Fsr();
    ~Fsr();
    Fsr(Fsr&&) noexcept;
    Fsr& operator=(Fsr&&) noexcept;

    [[nodiscard]] std::expected<void, Core::Error> resize(NoorRHI::Device& device, std::uint32_t renderWidth, std::uint32_t renderHeight,
                                                          std::uint32_t displayWidth, std::uint32_t displayHeight);
    [[nodiscard]] std::expected<void, Core::Error> record(NoorRHI::Device& device, NoorRHI::CommandList& list, PipelineCache& pipelines,
                                                          const UpscaleFrame& frame);
    //! The sample position in [0, 1) of a frame in the jitter sequence for the current sizes.
    NoorRHI::float2 getJitter(std::uint32_t frameIndex) const;

    //! The inputs from the denoise kernels: HDR colour, near / view depth (0 for the sky), and the motion to the previous frame in pixels.
    const NoorRHI::Image& getColor() const { return color; }
    const NoorRHI::Image& getDepth() const { return depth; }
    const NoorRHI::Image& getMotion() const { return motion; }
    const NoorRHI::Image& getOutput() const { return output; }
    std::uint32_t getRenderWidth() const { return renderWidth; }
    std::uint32_t getRenderHeight() const { return renderHeight; }

    struct Backend;

private:
    std::unique_ptr<Backend> backend;
    NoorRHI::Image color;
    NoorRHI::Image depth;
    NoorRHI::Image motion;
    NoorRHI::Image output;
    NoorRHI::Image dilatedDepth;
    NoorRHI::Image dilatedMotion;
    std::uint32_t renderWidth = 0;
    std::uint32_t renderHeight = 0;
    std::uint32_t displayWidth = 0;
    std::uint32_t displayHeight = 0;
};

}
