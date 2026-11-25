#pragma once

#include <cstdint>
#include <expected>

#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Training {

enum class LossKind : std::uint8_t {
    //! 0.8 L1 + 0.2 (1 - SSIM), with an 11x11 Gaussian window of sigma 1.5, as in 3DGS.
    L1Ssim,
    L2,
};

//! The image loss that every method shares, on the GPU, so that all methods optimize the same objective.
class Loss {
public:
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, std::uint32_t maxWidth, std::uint32_t maxHeight, LossKind kind);
    //! Records the loss of the RGB float image against the RGBA8 target, writes dLoss/dImage to the gradient and adds the loss to the value buffer.
    void record(NoorRHI::CommandList& list, NoorRHI::Ptr<float> image, NoorRHI::Ptr<NoorRHI::uint> target, NoorRHI::Ptr<float> gradient, std::uint32_t width,
                std::uint32_t height);
    //! Records the download of the loss value; read() returns it after the list completes.
    void recordReadback(NoorRHI::CommandList& list);
    [[nodiscard]] std::expected<double, NoorRHI::Error> read();
    LossKind getKind() const { return kind; }

private:
    NoorRHI::Device* device = nullptr;
    LossKind kind = LossKind::L1Ssim;
    NoorRHI::ComputePipeline filterMomentRows;
    NoorRHI::ComputePipeline filterMomentColumns;
    NoorRHI::ComputePipeline filterPartialRows;
    NoorRHI::ComputePipeline filterPartialColumns;
    NoorRHI::ComputePipeline squaredError;
    NoorRHI::ComputePipeline sumLosses;
    NoorRHI::Buffer<float> rows;
    NoorRHI::Buffer<float> partials;
    NoorRHI::Buffer<float> groupLosses;
    NoorRHI::Buffer<float> value;
    NoorRHI::Readback<float> readback;
    std::uint32_t maxWidth = 0;
    std::uint32_t maxHeight = 0;
    //! True while the value holds the sum of the records since the last readback.
    bool accumulating = false;
    bool oversized = false;
};

}
