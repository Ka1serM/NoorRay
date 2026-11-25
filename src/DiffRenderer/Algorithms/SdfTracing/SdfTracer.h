#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "DiffRenderer/Algorithms/SdfTracing/SdfScene.h"
#include "DiffRenderer/Algorithms/SdfTracing/Shaders/SphereTracing.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Algorithms {

struct SdfSampling {
    std::uint32_t samples = 8;
    std::uint32_t seed = 0;
    bool boundary = true;
};

//! Sphere tracing of a grid SDF from all views into one mosaic image; the backward pass adds silhouette edge sampling along scanlines.
class SdfTracer {
public:
    static constexpr std::uint32_t tileSize = 128;
    static constexpr std::uint32_t columns = 3;
    static constexpr std::uint32_t width = tileSize * columns;
    static constexpr std::uint32_t height = tileSize * static_cast<std::uint32_t>(SdfScene::viewCount) / columns;

    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const SdfScene& scene);
    [[nodiscard]] std::expected<void, NoorRHI::Error> render(std::span<const float> grid, const SdfSampling& sampling, std::span<float> result);
    //! Back-propagates the image gradient at the grid of the last render.
    [[nodiscard]] std::expected<void, NoorRHI::Error> backward(std::span<const float> imageGradient, const SdfSampling& sampling,
                                                               std::span<float> gridGradient);

private:
    [[nodiscard]] std::expected<void, NoorRHI::Error> submit(NoorRHI::CommandList&& list);
    SphereTracing::Arguments getArguments(std::size_t view, const SdfSampling& sampling) const;

    NoorRHI::Device* device = nullptr;
    NoorRHI::ComputePipeline renderPipeline;
    NoorRHI::ComputePipeline interiorPipeline;
    NoorRHI::ComputePipeline boundaryPipeline;
    NoorRHI::Buffer<float> parameterBuffer;
    NoorRHI::Buffer<float> weightBuffer;
    NoorRHI::Buffer<float> imageBuffer;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> gradientBuffer;
    NoorRHI::Readback<float> imageReadback;
    NoorRHI::Readback<NoorRHI::Atomic<float>> gradientReadback;
    std::vector<NoorRHI::Atomic<float>> zeroGradients;
    std::array<SphereTracing::Arguments, SdfScene::viewCount> views{};
};

}
