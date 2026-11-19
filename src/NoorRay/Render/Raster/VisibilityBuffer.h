#pragma once

#include <cstdint>
#include <expected>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Pipelines/Pipeline.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Gpu/GpuScene.h"
#include "NoorRay/Shaders/Common/CameraRecords.h"
#include "NoorRay/Shaders/Wavefront/WavefrontRecords.h"

namespace NoorRay::Render {

//! Rasterizes the triangle id of every pixel at the jittered sample position. The wavefront kernels turn the id into the hit of the primary ray.
class VisibilityBuffer {
public:
    [[nodiscard]] std::expected<void, Core::Error> resize(NoorRHI::Device& device, std::uint32_t width, std::uint32_t height);
    [[nodiscard]] std::expected<void, Core::Error> record(NoorRHI::Device& device, NoorRHI::CommandList& list, const Gpu::GpuScene& scene,
                                                          const Shaders::CameraRecord& camera, NoorRHI::float2 jitter);
    Shaders::RasterRecord getRecord(const Gpu::GpuScene& scene, NoorRHI::float2 jitter, bool rgb) const;
    const NoorRHI::Image& getDepth() const { return depth; }

private:
    NoorRHI::GraphicsPipeline pipeline;
    NoorRHI::Image ids;
    NoorRHI::Image depth;
    NoorRHI::Buffer<Shaders::CameraRecord> cameraBuffer;
};

}
