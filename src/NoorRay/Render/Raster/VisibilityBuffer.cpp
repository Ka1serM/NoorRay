#include "NoorRay/Render/Raster/VisibilityBuffer.h"

#include "NoorRay/Shaders/Raster/RasterRecords.h"
#include "NoorRayShadersVariants.h"

namespace NoorRay::Render {

std::expected<void, Core::Error> VisibilityBuffer::resize(NoorRHI::Device& device, const std::uint32_t width, const std::uint32_t height) {
    if (ids && ids.getWidth() == width && ids.getHeight() == height)
        return {};
    const NoorRHI::ImageDesc idDesc{
        .width = width,
        .height = height,
        .format = NoorRHI::ImageFormat::R32Uint,
        .usage = NoorRHI::ImageUsage::ColorAttachment | NoorRHI::ImageUsage::Sampled,
        .name = "NoorRay visibility",
    };
    if (auto opened = Core::check(ids.open(device, idDesc)); !opened)
        return opened;
    const NoorRHI::ImageDesc depthDesc{
        .width = width,
        .height = height,
        .format = NoorRHI::ImageFormat::D32Float,
        .usage = NoorRHI::ImageUsage::DepthAttachment,
        .name = "NoorRay visibility depth",
    };
    return Core::check(depth.open(device, depthDesc));
}

std::expected<void, Core::Error> VisibilityBuffer::record(NoorRHI::Device& device, NoorRHI::CommandList& list, const Gpu::GpuScene& scene,
                                                          const Shaders::CameraRecord& camera, const NoorRHI::float2 jitter) {
    if (!ids)
        return Core::fail(Core::ErrorCode::InvalidState, "the visibility buffer has no size");
    if (!pipeline) {
        const NoorRHI::GraphicsPipelineDesc desc{
            .vertex = *Shaders::visibilityVertex[0],
            .fragment = *Shaders::visibilityFragment[0],
            .state = {.cull = NoorRHI::CullMode::None, .depthCompare = NoorRHI::CompareOp::Less, .depthTest = true, .depthWrite = true},
            .colorFormat = NoorRHI::ImageFormat::R32Uint,
            .depthFormat = NoorRHI::ImageFormat::D32Float,
        };
        if (auto opened = Core::check(pipeline.open(device, desc)); !opened)
            return opened;
    }
    if (!cameraBuffer)
        if (auto opened = Core::check(cameraBuffer.open(device, 1)); !opened)
            return opened;
    list.upload(cameraBuffer, std::span<const Shaders::CameraRecord>(&camera, 1));
    list.pushLabel("VisibilityBuffer");
    list.beginRender({.color = &ids, .depth = &depth, .clear = true, .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}, .clearDepth = 1.0f});
    for (const Gpu::RasterDraw& draw : scene.getRasterDraws())
        list.draw(pipeline, draw.vertexCount, 1,
                  Shaders::VisibilityArguments{scene.getRecord(), cameraBuffer.getPtr(), jitter, draw.instance, draw.firstTriangle});
    list.endRender();
    list.popLabel();
    return {};
}

Shaders::RasterRecord VisibilityBuffer::getRecord(const Gpu::GpuScene& scene, const NoorRHI::float2 jitter, const bool rgb) const {
    return {
        .visibility = ids.getSampledHandle(),
        .enabled = 1,
        .jitter = jitter,
        .draws = scene.getRasterDrawRecords(),
        .drawCount = static_cast<std::uint32_t>(scene.getRasterDraws().size()),
        .rgb = rgb ? 1u : 0u,
    };
}

}
