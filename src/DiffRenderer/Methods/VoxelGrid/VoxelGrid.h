#pragma once

#include <array>
#include <cstdint>
#include <expected>

#include "DiffRenderer/Methods/Method.h"
#include "DiffRenderer/Training/GpuAdam.h"
#include "DiffRenderer/Training/PrefixScan.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Methods {

//! Plenoxels (Fridovich-Keil et al. 2022): density and SH degree 2 on a sparse voxel grid, coarse to fine, without a network.
class VoxelGrid final : public Method {
public:
    static constexpr std::size_t maxStages = 3;

    const Description& getDescription() const override;
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Scenes::Dataset& dataset, const Settings& settings) override;
    [[nodiscard]] std::expected<double, NoorRHI::Error> step(const Target& target, Training::Loss& loss) override;
    [[nodiscard]] std::expected<void, NoorRHI::Error> render(const Scenes::Camera& camera, Core::Image& image) override;
    Statistics getStatistics() const override;
    void close() override;

private:
    struct Stage {
        std::uint32_t resolution = 0;
        //! The share of the iterations before the stage starts.
        double start = 0.0;
    };

    struct Configuration {
        std::array<Stage, maxStages> stages{};
        std::size_t stageCount = 0;
        float densityVariation = 0.0f;
        float shVariation = 0.0f;
        //! The smallest render weight that keeps a voxel at an upsampling, times the resolution.
        float weightThreshold = 0.0f;
    };

    enum class Trace : std::uint8_t { Evaluate, Train, TrainAndWeigh };

    struct Grid {
        NoorRHI::Buffer<std::int32_t> links;
        //! The densities of all voxels, then their SH coefficients.
        NoorRHI::Buffer<float> parameters;
        NoorRHI::Buffer<NoorRHI::uint> occupancy;
        std::uint32_t resolution = 0;
        std::uint32_t count = 0;
    };

    [[nodiscard]] std::expected<void, NoorRHI::Error> submit(NoorRHI::CommandList&& list);
    [[nodiscard]] std::expected<void, NoorRHI::Error> resample(std::uint32_t resolution);
    [[nodiscard]] std::expected<std::uint32_t, NoorRHI::Error> countVoxels(std::uint32_t resolution, NoorRHI::Ptr<NoorRHI::uint> keep);
    void recordKeep(NoorRHI::CommandList& list, float threshold, NoorRHI::Ptr<NoorRHI::uint> keep, NoorRHI::Ptr<NoorRHI::uint> spare) const;
    void recordFill(NoorRHI::CommandList& list, std::uint64_t address, std::size_t count, NoorRHI::uint value) const;
    void recordTrace(NoorRHI::CommandList& list, const NoorRHI::ComputePipeline& pipeline, const Scenes::Camera& camera, Trace trace) const;
    void recordOptimizer(NoorRHI::CommandList& list);
    double getProgress() const;

    NoorRHI::Device* device = nullptr;
    const Scenes::Dataset* dataset = nullptr;
    Settings settings;
    Configuration configuration;
    NoorRHI::float4 placement{};
    bool contracted = false;
    NoorRHI::ComputePipeline renderPipeline;
    NoorRHI::ComputePipeline backwardPipeline;
    NoorRHI::ComputePipeline fillPipeline;
    NoorRHI::ComputePipeline variationPipeline;
    NoorRHI::ComputePipeline keepPipeline;
    NoorRHI::ComputePipeline dilatePipeline;
    NoorRHI::ComputePipeline countPipeline;
    NoorRHI::ComputePipeline linkPipeline;
    NoorRHI::ComputePipeline resamplePipeline;
    NoorRHI::ComputePipeline occupancyPipeline;
    Training::GpuAdam adam;
    Training::PrefixScan scan;
    Grid grid;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> gradients;
    NoorRHI::Buffer<float> first;
    NoorRHI::Buffer<float> second;
    NoorRHI::Buffer<NoorRHI::Atomic<NoorRHI::uint>> maxWeights;
    NoorRHI::Buffer<NoorRHI::uint> brickCounts;
    NoorRHI::Buffer<NoorRHI::uint> brickOffsets;
    NoorRHI::Readback<NoorRHI::uint> lastCount;
    NoorRHI::Readback<NoorRHI::uint> lastOffset;
    NoorRHI::Buffer<float> image;
    NoorRHI::Buffer<float> imageGradients;
    NoorRHI::Readback<float> imageReadback;
    std::array<int, maxStages> stageStarts{};
    std::size_t stage = 0;
    int iteration = 0;
    std::uint32_t optimizerStep = 0;
};

}
