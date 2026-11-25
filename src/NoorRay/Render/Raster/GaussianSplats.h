#pragma once

#include <cstdint>
#include <expected>
#include <memory>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Pipelines/Pipeline.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Gpu/GpuScene.h"
#include "NoorRay/Render/PipelineCache.h"
#include "NoorRay/Shaders/Common/CameraRecords.h"
#include "NoorRay/Shaders/Raster/SplatRecords.h"

namespace NoorRay::Render {

//! Draws the Gaussians as 3DGS splats, compacted, radix sorted back to front and blended over the visibility depth; an unchanged view only redraws.
class GaussianSplats {
public:
    static constexpr NoorRHI::ImageFormat format = NoorRHI::ImageFormat::Rgba16Float;

    [[nodiscard]] std::expected<void, Core::Error> resize(NoorRHI::Device& device, std::uint32_t width, std::uint32_t height);
    //! depth is the visibility depth drawn with the same jitter, so meshes in front hide the splats.
    [[nodiscard]] std::expected<void, Core::Error> record(NoorRHI::Device& device, NoorRHI::CommandList& list, PipelineCache& pipelines,
                                                          const Gpu::GpuScene& scene, const Shaders::CameraRecord& camera, const NoorRHI::Image& depth,
                                                          NoorRHI::float2 jitter);

    //! Premultiplied linear radiance with coverage in alpha.
    const NoorRHI::Image& getImage() const { return image; }

private:
    [[nodiscard]] std::expected<void, Core::Error> reserve(NoorRHI::Device& device, std::uint32_t count);
    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device);
    [[nodiscard]] std::expected<void, Core::Error> sort(NoorRHI::Device& device, NoorRHI::CommandList& list, PipelineCache& pipelines,
                                                        Shaders::SplatArguments arguments);

    NoorRHI::GraphicsPipeline pipeline;
    NoorRHI::Image image;
    NoorRHI::Buffer<Shaders::CameraRecord> cameraBuffer;
    NoorRHI::Buffer<std::uint32_t> keys;
    NoorRHI::Buffer<std::uint32_t> values;
    NoorRHI::Buffer<std::uint32_t> sortedKeys;
    NoorRHI::Buffer<std::uint32_t> sortedValues;
    NoorRHI::Buffer<std::uint32_t> histogram;
    NoorRHI::Buffer<std::uint32_t> partials;
    NoorRHI::Buffer<NoorRHI::uint4> splats;
    NoorRHI::Buffer<std::uint32_t> state;
    NoorRHI::Buffer<NoorRHI::DispatchSize> dispatches;
    NoorRHI::Buffer<NoorRHI::DrawIndexedArgs> draw;
    NoorRHI::Buffer<NoorRHI::Atomic<std::uint32_t>> counters;
    NoorRHI::Buffer<std::uint32_t> quadIndices;
    std::uint32_t capacity = 0;
    bool sorted = false;
    Shaders::CameraRecord sortedCamera{};
};

}
