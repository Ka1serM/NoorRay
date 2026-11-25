#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <span>

#include "DiffRenderer/Algorithms/PathTracing/PathScene.h"
#include "DiffRenderer/Algorithms/PathTracing/Shaders/DirectLighting.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Algorithms {

struct PathSampling {
    std::uint32_t samples = 16;
    std::uint32_t lightSamples = 1;
    std::uint32_t edgeSamples = 512;
    std::uint32_t seed = 0;
    bool boundary = true;
};

//! Direct illumination of triangle meshes with ray queries; the backward pass adds edge sampling of the primary and the shadow silhouettes.
class PathTracer {
public:
    static constexpr std::uint32_t width = 256;
    static constexpr std::uint32_t height = 256;

    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const PathScene& scene);
    [[nodiscard]] std::expected<void, NoorRHI::Error> render(const PathParameters& parameters, const PathSampling& sampling, std::span<float> result);
    //! Back-propagates the image gradient at the parameters of the last render.
    [[nodiscard]] std::expected<void, NoorRHI::Error> backward(std::span<const float> imageGradient, const PathSampling& sampling, PathParameters& result);

private:
    [[nodiscard]] std::expected<void, NoorRHI::Error> submit(NoorRHI::CommandList&& list);

    NoorRHI::Device* device = nullptr;
    NoorRHI::ComputePipeline renderPipeline;
    NoorRHI::ComputePipeline gradientPipeline;
    NoorRHI::ComputePipeline boundaryPipeline;
    NoorRHI::Buffer<NoorRHI::float3> groundPositions;
    NoorRHI::Buffer<std::uint32_t> groundIndices;
    NoorRHI::Buffer<NoorRHI::float3> objectPositions;
    NoorRHI::Buffer<std::uint32_t> objectIndices;
    NoorRHI::Buffer<NoorRHI::float3> objectNormals;
    NoorRHI::Buffer<NoorRHI::uint4> objectEdges;
    NoorRHI::Buffer<float> parameterBuffer;
    NoorRHI::Buffer<float> weightBuffer;
    NoorRHI::Buffer<float> imageBuffer;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> gradientBuffer;
    NoorRHI::Readback<float> imageReadback;
    NoorRHI::Readback<NoorRHI::Atomic<float>> gradientReadback;
    NoorRHI::AccelerationStructure groundBlas;
    NoorRHI::AccelerationStructure objectBlas;
    NoorRHI::AccelerationStructure tlas;
    std::array<NoorRHI::Instance, 2> instances;
    DirectLighting::Arguments arguments{};
};

}
