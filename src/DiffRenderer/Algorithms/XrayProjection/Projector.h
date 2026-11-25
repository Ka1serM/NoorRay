#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "DiffRenderer/Algorithms/VolumeRendering/Phantom.h"
#include "DiffRenderer/Algorithms/XrayProjection/Shaders/Radiograph.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Algorithms {

//! ZXY Euler angles, then the translation of the source.
using Pose = std::array<float, 6>;

//! The digitally reconstructed radiograph of the phantom in the geometry of DiffDRR: one line integral of the trilinear density for each detector pixel.
class Projector {
public:
    static constexpr std::uint32_t imageSize = 256;

    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Phantom& phantom);
    [[nodiscard]] std::expected<void, NoorRHI::Error> render(const Pose& pose, std::span<float> lineIntegrals);
    //! Takes the gradient of the loss with respect to each line integral of the last pose and returns it with respect to the pose.
    [[nodiscard]] std::expected<void, NoorRHI::Error> backward(std::span<const float> lineIntegralGradient, Pose& poseGradient);

private:
    [[nodiscard]] std::expected<void, NoorRHI::Error> submit(NoorRHI::CommandList&& list);

    NoorRHI::Device* device = nullptr;
    NoorRHI::ComputePipeline renderPipeline;
    NoorRHI::ComputePipeline backwardPipeline;
    NoorRHI::Buffer<float> density;
    NoorRHI::Buffer<float> image;
    NoorRHI::Buffer<float> imageGradients;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> poseGradients;
    NoorRHI::Readback<float> imageReadback;
    NoorRHI::Readback<NoorRHI::Atomic<float>> poseReadback;
    Radiograph::Arguments arguments{};
};

}
