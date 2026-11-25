#pragma once

#include <cstdint>
#include <expected>

#include "DiffRenderer/Methods/GaussianSplatting/GaussianModel.h"
#include "DiffRenderer/Methods/GaussianSplatting/Shaders/TileSplatting.h"
#include "DiffRenderer/Methods/Method.h"
#include "DiffRenderer/Training/PrefixScan.h"
#include "DiffRenderer/Training/RadixSort.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Methods {

//! 3D Gaussian Splatting as in Kerbl et al. 2023 and its CUDA rasterizer diff-gaussian-rasterization.
class GaussianSplatting final : public Method {
public:
    const Description& getDescription() const override;
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Scenes::Dataset& dataset, const Settings& settings) override;
    [[nodiscard]] std::expected<double, NoorRHI::Error> step(const Target& target, Training::Loss& loss) override;
    [[nodiscard]] std::expected<void, NoorRHI::Error> render(const Scenes::Camera& camera, Core::Image& image) override;
    Statistics getStatistics() const override { return model.getStatistics(); }
    //! The SH of 3DGS were trained by alpha blending in the sRGB encoding, so a renderer that blends in linear radiance sees them a little differently.
    [[nodiscard]] std::expected<std::optional<TrainedGaussians>, NoorRHI::Error> readGaussians() override;
    void close() override { *this = GaussianSplatting(); }

private:
    [[nodiscard]] std::expected<void, NoorRHI::Error> openPipelines();
    [[nodiscard]] std::expected<void, NoorRHI::Error> openWorkBuffers(std::uint32_t capacity);
    [[nodiscard]] std::expected<void, NoorRHI::Error> openInstanceBuffers(std::uint32_t capacity);
    [[nodiscard]] std::expected<void, NoorRHI::Error> submit(NoorRHI::CommandList&& list) const;
    [[nodiscard]] std::expected<void, NoorRHI::Error> recordForward(NoorRHI::CommandList& list, const Scenes::Camera& camera, bool accumulate);
    void recordBackward(NoorRHI::CommandList& list) const;
    [[nodiscard]] std::expected<void, NoorRHI::Error> readInstanceTotal();
    TileSplatting::Parameters getPointers(NoorRHI::Buffer<float> GaussianModel::GroupBuffers::* member) const;

    NoorRHI::Device* device = nullptr;
    GaussianModel model;
    NoorRHI::float4 background{};
    std::uint32_t workCapacity = 0;
    std::uint32_t instanceCapacity = 0;
    std::uint32_t instanceTotal = 0;
    std::uint32_t maxPixels = 0;
    std::uint32_t maxTiles = 0;

    NoorRHI::ComputePipeline preprocess;
    NoorRHI::ComputePipeline countInstances;
    NoorRHI::ComputePipeline duplicateKeys;
    NoorRHI::ComputePipeline clearRanges;
    NoorRHI::ComputePipeline identifyRanges;
    NoorRHI::ComputePipeline blend;
    NoorRHI::ComputePipeline blendBackward;
    NoorRHI::ComputePipeline preprocessBackward;
    Training::PrefixScan scan;
    Training::RadixSort sort;

    NoorRHI::Buffer<NoorRHI::float4> projected;
    NoorRHI::Buffer<NoorRHI::float4> conics;
    NoorRHI::Buffer<float> colors;
    NoorRHI::Buffer<NoorRHI::uint> tilesTouched;
    NoorRHI::Buffer<NoorRHI::uint> offsets;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> splatGradients;

    NoorRHI::Buffer<std::uint64_t> keys;
    NoorRHI::Buffer<NoorRHI::uint> values;
    NoorRHI::Buffer<NoorRHI::uint> instanceCounts;
    NoorRHI::Readback<NoorRHI::uint> instanceTotalReadback;
    NoorRHI::Buffer<NoorRHI::uint> ranges;
    NoorRHI::Buffer<float> image;
    NoorRHI::Buffer<float> finalTransmittance;
    NoorRHI::Buffer<NoorRHI::uint> lastContributors;
    NoorRHI::Buffer<float> imageGradient;
    NoorRHI::Readback<float> imageReadback;
    TileSplatting::Arguments arguments{};
};

}
