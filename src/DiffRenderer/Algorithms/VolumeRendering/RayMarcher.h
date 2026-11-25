#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "DiffRenderer/Algorithms/VolumeRendering/Phantom.h"
#include "DiffRenderer/Algorithms/VolumeRendering/Shaders/RayMarching.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Algorithms {

//! Emission-absorption ray marching with a piecewise-linear RGBA transfer function; the backward pass inverts the compositing front to back (DiffDVR).
class RayMarcher {
public:
    static constexpr std::uint32_t viewSize = 256;
    //! The four views are the tiles of one 2x2 mosaic.
    static constexpr std::uint32_t mosaicSize = 2 * viewSize;
    static constexpr std::uint32_t transferFunctionSize = 32;

    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Phantom& phantom);
    //! Renders the transfer function, four floats for each entry, into the RGB mosaic.
    [[nodiscard]] std::expected<void, NoorRHI::Error> render(std::span<const float> transferFunction, std::span<float> mosaic);
    //! Takes the gradient of the loss with respect to the RGB mosaic of the last render and returns it with respect to each transfer function value.
    [[nodiscard]] std::expected<void, NoorRHI::Error> backward(std::span<const float> mosaicGradient, std::span<float> transferFunctionGradient);

private:
    [[nodiscard]] std::expected<void, NoorRHI::Error> submit(NoorRHI::CommandList&& list);
    static std::size_t getMosaicIndex(std::size_t pixel);

    NoorRHI::Device* device = nullptr;
    NoorRHI::ComputePipeline renderPipeline;
    NoorRHI::ComputePipeline backwardPipeline;
    NoorRHI::Buffer<float> density;
    NoorRHI::Buffer<float> tumor;
    NoorRHI::Buffer<NoorRHI::float4> transferFunction;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> transferFunctionGradients;
    NoorRHI::Buffer<NoorRHI::float4> image;
    NoorRHI::Buffer<NoorRHI::float4> imageGradients;
    NoorRHI::Readback<NoorRHI::float4> imageReadback;
    NoorRHI::Readback<NoorRHI::Atomic<float>> gradientReadback;
    std::vector<NoorRHI::float4> stagedGradients;
    std::vector<NoorRHI::Atomic<float>> zeros;
    RayMarching::Arguments arguments{};
};

}
