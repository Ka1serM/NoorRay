#pragma once

#include <array>
#include <expected>
#include <span>

#include "DiffRenderer/Algorithms/GaussianSplatting/Gaussians.h"
#include "NoorRHI/NoorRHI.h"
#include "DiffRenderer/Algorithms/GaussianSplatting/Shaders/Splatting.h"

namespace DiffRenderer::Algorithms {

struct SplatCamera {
    std::array<NoorRHI::float4, 3> view{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 4.0f}}};
    NoorRHI::float4 intrinsics{300.0f, 300.0f, 128.0f, 128.0f};
    NoorRHI::float4 background{0.1f, 0.15f, 0.25f, 0.0f};
    std::uint32_t width = 256;
    std::uint32_t height = 256;
    float nearPlane = 0.01f;
    float farPlane = 1e10f;
    float blur = 0.3f;
};

//! The forward and backward passes of 3D Gaussian splatting, as in gsplat: project, sort by depth, composite front to back.
class Splatter {
public:
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, std::size_t count, const SplatCamera& camera);
    [[nodiscard]] std::expected<void, NoorRHI::Error> render(const Gaussians& gaussians, std::span<float> image);
    [[nodiscard]] std::expected<void, NoorRHI::Error> backward(std::span<const float> imageGradient, Gaussians& gradients);

private:
    struct Stage {
        const NoorRHI::ComputePipeline* pipeline;
        NoorRHI::DispatchSize size;
    };

    [[nodiscard]] std::expected<void, NoorRHI::Error> submit(NoorRHI::CommandList&& list);
    void record(NoorRHI::CommandList& list, std::span<const Stage> stages) const;

    NoorRHI::Device* device = nullptr;
    NoorRHI::ComputePipeline project;
    NoorRHI::ComputePipeline sort;
    NoorRHI::ComputePipeline composite;
    NoorRHI::ComputePipeline compositeBackward;
    NoorRHI::ComputePipeline projectBackward;
    std::array<NoorRHI::Buffer<float>, 5> parameters;
    std::array<NoorRHI::Buffer<float>, 5> gradients;
    NoorRHI::Buffer<NoorRHI::float4> projected;
    NoorRHI::Buffer<NoorRHI::uint> order;
    NoorRHI::Buffer<float> image;
    NoorRHI::Buffer<float> transmittance;
    NoorRHI::Buffer<NoorRHI::uint> contributors;
    NoorRHI::Buffer<float> imageGradient;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> projectedGradient;
    NoorRHI::Readback<float> imageReadback;
    std::array<NoorRHI::Readback<float>, 5> gradientReadbacks;
    Splatting::Arguments arguments{};
};

}
