#pragma once

#include <array>
#include <cstdint>
#include <expected>

#include "DiffRenderer/Methods/GaussianSplatting/Shaders/TileSplatting.h"
#include "DiffRenderer/Methods/Method.h"
#include "DiffRenderer/Training/GpuAdam.h"
#include "DiffRenderer/Training/PrefixScan.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Methods {

//! The Gaussians of 3DGS with their Adam state, schedules, densification and opacity reset, as Kerbl et al. 2023 train them for any renderer.
class GaussianModel {
public:
    enum Group : std::uint8_t { Means, FeaturesDc, FeaturesRest, Opacities, Scales, Rotations, GroupCount };
    static constexpr std::array<std::uint32_t, GroupCount> strides{3, 3, 45, 1, 3, 4};
    static constexpr std::uint32_t maxThreads = 65535u * 256u;

    struct GroupBuffers {
        NoorRHI::Buffer<float> parameters;
        NoorRHI::Buffer<float> first;
        NoorRHI::Buffer<float> second;
        //! Work memory of each step, so it is not gathered when the densification changes the Gaussians.
        NoorRHI::Buffer<float> gradients;
    };

    struct Phase {
        bool densifying = false;
        bool densify = false;
        bool reset = false;
    };

    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Scenes::Dataset& dataset, const Settings& settings);
    //! Starts an iteration and sets the Adam step of each group that it updates.
    Phase beginStep();
    void recordUpdate(NoorRHI::CommandList& list) const;
    //! Densifies and resets the opacities as the phase asks, after the update of the iteration completed; the capacity can grow.
    [[nodiscard]] std::expected<void, NoorRHI::Error> finishStep(const Phase& phase);
    [[nodiscard]] std::expected<TrainedGaussians, NoorRHI::Error> readGaussians() const;

    const GroupBuffers& getGroup(Group group) const { return groups[group]; }
    NoorRHI::Ptr<float> getGradientSums() const { return gradientSums.getPtr(); }
    NoorRHI::Ptr<float> getGradientCounts() const { return gradientCounts.getPtr(); }
    std::uint32_t getCount() const { return count; }
    std::uint32_t getCapacity() const { return capacity; }
    std::uint32_t getShDegree() const;
    Statistics getStatistics() const;
    static std::uint32_t getThreads(std::uint64_t count);

private:
    struct Schedule {
        int iterations = 30000;
        int shInterval = 1000;
        int densifyFrom = 500;
        int densifyUntil = 15000;
        int densifyInterval = 100;
        int opacityResetInterval = 3000;
    };

    [[nodiscard]] std::expected<void, NoorRHI::Error> openWorkBuffers(std::uint32_t capacity);
    [[nodiscard]] std::expected<void, NoorRHI::Error> upload(const Scenes::Dataset& dataset);
    [[nodiscard]] std::expected<void, NoorRHI::Error> submit(NoorRHI::CommandList&& list) const;
    void setOptimizer(bool opacity);
    [[nodiscard]] std::expected<void, NoorRHI::Error> densify();
    [[nodiscard]] std::expected<void, NoorRHI::Error> gather(NoorRHI::Buffer<float>& buffer, const NoorRHI::Buffer<NoorRHI::uint>& sources, std::uint32_t total,
                                                             std::uint32_t stride);
    [[nodiscard]] std::expected<void, NoorRHI::Error> resetOpacity() const;
    void recordClear(NoorRHI::CommandList& list, NoorRHI::Ptr<float> destination, std::uint32_t count) const;
    TileSplatting::DensifyArguments getDensifyArguments() const;
    float getPositionLearningRate() const;

    NoorRHI::Device* device = nullptr;
    Schedule schedule;
    Settings settings;
    float extent = 1.0f;
    bool whiteBackground = false;
    int iteration = 0;
    std::uint32_t adamSteps = 0;
    std::uint32_t opacityAdamSteps = 0;
    std::uint32_t densifications = 0;
    std::uint32_t count = 0;
    std::uint32_t capacity = 0;

    NoorRHI::ComputePipeline countOutputs;
    NoorRHI::ComputePipeline totalOutputs;
    NoorRHI::ComputePipeline writeSources;
    NoorRHI::ComputePipeline splitChildren;
    NoorRHI::ComputePipeline clearFloats;
    NoorRHI::ComputePipeline resetOpacities;
    Training::GpuAdam adam;
    Training::PrefixScan scan;

    std::array<GroupBuffers, GroupCount> groups;
    //! The Adam step of each group with its first bias correction, and its second bias correction; zero holds the group.
    std::array<float, GroupCount> stepSizes{};
    std::array<float, GroupCount> secondCorrections{};
    NoorRHI::Buffer<float> gradientSums;
    NoorRHI::Buffer<float> gradientCounts;
    NoorRHI::Buffer<NoorRHI::uint> outputs;
    NoorRHI::Buffer<NoorRHI::uint> offsets;
    NoorRHI::Buffer<NoorRHI::uint> parameterSources;
    NoorRHI::Buffer<NoorRHI::uint> momentSources;
    NoorRHI::Buffer<NoorRHI::uint> children;
    NoorRHI::Buffer<NoorRHI::uint> newCount;
    NoorRHI::Readback<NoorRHI::uint> newCountReadback;
};

}
