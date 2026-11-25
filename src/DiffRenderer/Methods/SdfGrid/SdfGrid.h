#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

#include "DiffRenderer/Methods/Method.h"
#include "DiffRenderer/Methods/SdfGrid/Shaders/SurfaceRendering.h"
#include "DiffRenderer/Training/GpuAdam.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Methods {

//! A parameter array with its gradient and the two Adam moments.
struct GridParameters {
    NoorRHI::Buffer<float> values;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> gradients;
    NoorRHI::Buffer<float> first;
    NoorRHI::Buffer<float> second;

    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, std::size_t count);
    Training::AdamGroup getGroup(float learningRate) const;
    NoorRHI::Ptr<float> getGradientPtr() const { return NoorRHI::Ptr<float>(gradients.getPtr().getAddress()); }
};

//! An SDF and SH colour voxel grid, sphere traced as an opaque surface, with the relaxed boundary term of Wang et al. 2024.
class SdfGrid final : public Method {
public:
    const Description& getDescription() const override;
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Scenes::Dataset& dataset, const Settings& settings) override;
    [[nodiscard]] std::expected<double, NoorRHI::Error> step(const Target& target, Training::Loss& loss) override;
    [[nodiscard]] std::expected<void, NoorRHI::Error> render(const Scenes::Camera& camera, Core::Image& image) override;
    Statistics getStatistics() const override;
    //! Keeps a copy of the SDF on the CPU for writeMesh.
    void close() override;
    //! Writes the zero level set of the last trained SDF as a binary PLY.
    [[nodiscard]] std::expected<void, std::string> writeMesh(const std::filesystem::path& path) const;

private:
    [[nodiscard]] std::expected<void, NoorRHI::Error> openGrid(const SurfaceRendering::Grid& grid, const GridParameters* sourceSdf,
                                                               const GridParameters* sourceColour, const SurfaceRendering::Grid& sourceGrid);
    [[nodiscard]] std::expected<void, NoorRHI::Error> refine(std::uint32_t edge);
    [[nodiscard]] std::expected<void, NoorRHI::Error> openImage(std::uint32_t width, std::uint32_t height);
    [[nodiscard]] std::expected<std::vector<float>, NoorRHI::Error> downloadSdf() const;
    [[nodiscard]] std::expected<void, NoorRHI::Error> submit(NoorRHI::CommandList&& list) const;
    SurfaceRendering::Arguments getArguments(const Scenes::Camera& camera, std::uint32_t samplesPerSide) const;
    void recordRedistance(NoorRHI::CommandList& list);
    void recordClear(NoorRHI::CommandList& list, NoorRHI::Ptr<float> target, std::size_t count) const;
    float getProgress() const;

    NoorRHI::Device* device = nullptr;
    Settings settings;
    Training::GpuAdam adam;
    NoorRHI::ComputePipeline renderPipeline;
    NoorRHI::ComputePipeline backwardPipeline;
    NoorRHI::ComputePipeline seedPipeline;
    NoorRHI::ComputePipeline floodPipeline;
    NoorRHI::ComputePipeline redistancePipeline;
    NoorRHI::ComputePipeline smoothPipeline;
    NoorRHI::ComputePipeline clearPipeline;
    NoorRHI::ComputePipeline resamplePipeline;
    SurfaceRendering::Grid grid{};
    GridParameters sdf;
    GridParameters colour;
    GridParameters environment;
    std::array<NoorRHI::Buffer<NoorRHI::float4>, 2> seeds;
    NoorRHI::Buffer<float> image;
    NoorRHI::Buffer<float> imageGradient;
    NoorRHI::Readback<float> imageReadback;
    std::uint32_t imageWidth = 0;
    std::uint32_t imageHeight = 0;
    NoorRHI::float3 boundsLower{};
    NoorRHI::float3 boundsUpper{};
    NoorRHI::float4 background{};
    bool cropping = false;
    std::uint32_t stage = 0;
    std::uint32_t iteration = 0;
    std::uint32_t adamStep = 0;
    std::vector<float> closedSdf;
    SurfaceRendering::Grid closedGrid{};
};

}
