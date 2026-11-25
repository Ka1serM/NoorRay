#pragma once

#include <expected>
#include <span>

#include "DiffRenderer/Algorithms/Rasterization/Shaders/Rasterizing.h"
#include "DiffRenderer/Algorithms/Rasterization/TexturedCube.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Algorithms {

//! The forward and backward passes of nvdiffrast: rasterize, interpolate, sample the texture, then antialias the silhouette edges analytically.
class Rasterizer {
public:
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const TexturedCube& cube, bool antialiasing);
    [[nodiscard]] std::expected<void, NoorRHI::Error> render(std::span<const NoorRHI::float4> clip, std::span<float> image);
    [[nodiscard]] std::expected<void, NoorRHI::Error> backward(std::span<const float> imageGradient, std::span<NoorRHI::float4> clipGradient);

private:
    void record(NoorRHI::CommandList& list, std::span<const NoorRHI::ComputePipeline* const> pipelines) const;
    [[nodiscard]] std::expected<void, NoorRHI::Error> submit(NoorRHI::CommandList&& list);

    NoorRHI::Device* device = nullptr;
    NoorRHI::ComputePipeline clear;
    NoorRHI::ComputePipeline rasterize;
    NoorRHI::ComputePipeline shade;
    NoorRHI::ComputePipeline antialias;
    NoorRHI::ComputePipeline antialiasBackward;
    NoorRHI::ComputePipeline shadeBackward;
    NoorRHI::Buffer<NoorRHI::float4> positions;
    NoorRHI::Buffer<NoorRHI::uint> triangles;
    NoorRHI::Buffer<int> opposites;
    NoorRHI::Buffer<NoorRHI::float2> uvs;
    NoorRHI::Buffer<NoorRHI::uint> uvTriangles;
    NoorRHI::Buffer<float> texture;
    NoorRHI::Buffer<NoorRHI::float4> raster;
    NoorRHI::Buffer<float> color;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> image;
    NoorRHI::Buffer<float> imageGradient;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> colorGradient;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> positionGradient;
    NoorRHI::Readback<NoorRHI::Atomic<float>> imageReadback;
    NoorRHI::Readback<NoorRHI::Atomic<float>> positionReadback;
    Rasterizing::Arguments arguments{};
    NoorRHI::uint clearCount = 0;
};

}
