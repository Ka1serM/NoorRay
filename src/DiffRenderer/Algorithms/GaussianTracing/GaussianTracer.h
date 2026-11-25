#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "DiffRenderer/Algorithms/GaussianSplatting/Gaussians.h"
#include "DiffRenderer/Algorithms/GaussianTracing/GaussianScene.h"
#include "DiffRenderer/Algorithms/GaussianTracing/Shaders/GaussianPaths.h"
#include "DiffRenderer/Core/GaussianProxies.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Algorithms {

enum class GaussianShading : std::uint8_t { Emissive, Scattering };

struct GaussianSampling {
    std::uint32_t samples = 16;
    std::uint32_t seed = 0;
    //! Renders four wavelengths for each sample, or else the sRGB channels.
    bool spectral = true;
    //! The box filter in pixels; zero samples the pixel centres.
    float filterWidth = 1.0f;
};

//! The backward pass must use another seed than the image of the loss, so that the gradient of a nonlinear loss stays unbiased.
class GaussianTracer {
public:
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const GaussianScene& scene, std::size_t count, std::uint32_t width,
                                                           std::uint32_t height);
    [[nodiscard]] std::expected<void, NoorRHI::Error> setParameters(const Gaussians& gaussians);
    [[nodiscard]] std::expected<void, NoorRHI::Error> render(GaussianShading shading, const GaussianSampling& sampling, std::span<float> result);
    [[nodiscard]] std::expected<void, NoorRHI::Error> backward(GaussianShading shading, const GaussianSampling& sampling, std::span<const float> imageGradient,
                                                               Gaussians& result);

private:
    [[nodiscard]] std::expected<void, NoorRHI::Error> openMesh(const GaussianScene& scene);
    void setScene(const GaussianScene& scene);
    [[nodiscard]] std::expected<void, NoorRHI::Error> submit(NoorRHI::CommandList&& list);
    [[nodiscard]] std::expected<void, NoorRHI::Error> dispatch(const NoorRHI::ComputePipeline& pipeline, const GaussianSampling& sampling,
                                                               std::uint32_t firstSample);

    NoorRHI::Device* device = nullptr;
    std::array<NoorRHI::ComputePipeline, 2> renderPipelines;
    std::array<NoorRHI::ComputePipeline, 2> gradientPipelines;
    Core::GaussianProxies proxies;
    NoorRHI::Buffer<NoorRHI::float3> positions;
    NoorRHI::Buffer<std::uint32_t> indices;
    NoorRHI::Buffer<NoorRHI::float3> albedos;
    NoorRHI::AccelerationStructure meshBlas;
    NoorRHI::AccelerationStructure meshTlas;
    NoorRHI::Buffer<NoorRHI::float4> spectralEntries;
    NoorRHI::Buffer<float> parameters;
    NoorRHI::Buffer<float> weights;
    NoorRHI::Buffer<float> image;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> gradients;
    NoorRHI::Readback<float> imageReadback;
    NoorRHI::Readback<NoorRHI::Atomic<float>> gradientReadback;
    std::vector<float> rotations;
    GaussianPaths::Arguments arguments{};
};

}
