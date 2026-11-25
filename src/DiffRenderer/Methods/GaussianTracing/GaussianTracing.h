#pragma once

#include <cstdint>
#include <expected>

#include "DiffRenderer/Core/GaussianProxies.h"
#include "DiffRenderer/Methods/GaussianSplatting/GaussianModel.h"
#include "DiffRenderer/Methods/GaussianTracing/Shaders/GaussianRays.h"
#include "DiffRenderer/Methods/Method.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Methods {

//! The Gaussians of 3DGS, ray traced and composited in linear radiance in the order of their peaks, as the NoorRay path tracer renders them.
class GaussianTracing final : public Method {
public:
    const Description& getDescription() const override;
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Scenes::Dataset& dataset, const Settings& settings) override;
    [[nodiscard]] std::expected<double, NoorRHI::Error> step(const Target& target, Training::Loss& loss) override;
    [[nodiscard]] std::expected<void, NoorRHI::Error> render(const Scenes::Camera& camera, Core::Image& image) override;
    Statistics getStatistics() const override { return model.getStatistics(); }
    [[nodiscard]] std::expected<std::optional<TrainedGaussians>, NoorRHI::Error> readGaussians() override;
    void close() override { *this = GaussianTracing(); }

private:
    [[nodiscard]] std::expected<void, NoorRHI::Error> openWorkBuffers(std::uint32_t capacity);
    [[nodiscard]] std::expected<void, NoorRHI::Error> submit(NoorRHI::CommandList&& list) const;
    [[nodiscard]] std::expected<void, NoorRHI::Error> rebuildProxies();
    [[nodiscard]] std::expected<GaussianRays::Arguments, NoorRHI::Error> getArguments(const Scenes::Camera& camera, bool accumulate) const;
    GaussianRays::Parameters getPointers(NoorRHI::Buffer<float> GaussianModel::GroupBuffers::* member) const;
    [[nodiscard]] std::expected<void, NoorRHI::Error> submitRows(const NoorRHI::ComputePipeline& pipeline, const GaussianRays::Arguments& arguments) const;

    NoorRHI::Device* device = nullptr;
    GaussianModel model;
    Core::GaussianProxies proxies;
    NoorRHI::float4 background{};
    std::uint32_t workCapacity = 0;
    std::uint32_t maxPixels = 0;

    NoorRHI::ComputePipeline clearGradients;
    NoorRHI::ComputePipeline traceForward;
    NoorRHI::ComputePipeline traceBackward;
    NoorRHI::ComputePipeline gatherGradients;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> rayGradients;
    NoorRHI::Buffer<float> image;
    NoorRHI::Buffer<float> radiance;
    NoorRHI::Buffer<float> imageGradient;
    NoorRHI::Readback<float> imageReadback;
};

}
