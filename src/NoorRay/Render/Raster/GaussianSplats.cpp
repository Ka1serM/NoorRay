#include "NoorRay/Render/Raster/GaussianSplats.h"

#include <array>
#include <cstring>
#include <utility>

#include "NoorRayShadersVariants.h"

namespace NoorRay::Render {

namespace {

constexpr std::array<std::uint32_t, 6> quadCorners{0, 1, 2, 1, 3, 2};

std::uint32_t getBlockCount(const std::uint32_t count) {
    return (count + Shaders::splatGroupSize - 1) / Shaders::splatGroupSize;
}

//! One thread per Gaussian, folded into rows of splatRowGroups blocks.
NoorRHI::DispatchSize getBlockDispatch(const std::uint32_t blocks) {
    const std::uint32_t rows = (blocks + Shaders::splatRowGroups - 1) / Shaders::splatRowGroups;
    return {std::min(blocks, Shaders::splatRowGroups) * Shaders::splatGroupSize, rows, 1};
}

//! The projection depends on the view and the image, not on the previous camera that reprojection reads.
bool sameView(const Shaders::CameraRecord& a, const Shaders::CameraRecord& b) {
    return std::memcmp(a.worldToCamera, b.worldToCamera, sizeof(a.worldToCamera)) == 0 && std::memcmp(&a.sensor, &b.sensor, sizeof(a.sensor)) == 0
           && std::memcmp(&a.projection, &b.projection, sizeof(a.projection)) == 0 && std::memcmp(&a.clip, &b.clip, sizeof(a.clip)) == 0
           && std::memcmp(&a.meta, &b.meta, sizeof(a.meta)) == 0;
}

std::expected<const NoorRHI::ComputePipeline*, Core::Error> getPipeline(NoorRHI::Device& device, PipelineCache& pipelines, const NoorRHI::Shader* shader) {
    return pipelines.get(device, *shader);
}

}

std::expected<void, Core::Error> GaussianSplats::resize(NoorRHI::Device& device, const std::uint32_t width, const std::uint32_t height) {
    if (image && image.getWidth() == width && image.getHeight() == height)
        return {};
    sorted = false;
    const NoorRHI::ImageDesc desc{
        .width = width,
        .height = height,
        .format = format,
        .usage = NoorRHI::ImageUsage::ColorAttachment | NoorRHI::ImageUsage::Sampled,
        .name = "NoorRay Gaussian splats",
    };
    return Core::check(image.open(device, desc));
}

std::expected<void, Core::Error> GaussianSplats::reserve(NoorRHI::Device& device, const std::uint32_t count) {
    if (count <= capacity)
        return {};
    sorted = false;
    for (NoorRHI::Buffer<std::uint32_t>* buffer : {&keys, &values, &sortedKeys, &sortedValues})
        if (auto opened = Core::check(buffer->open(device, count)); !opened)
            return opened;
    const std::size_t histogramSize = std::size_t{Shaders::splatDigits} * getBlockCount(count);
    if (auto opened = Core::check(histogram.open(device, histogramSize)); !opened)
        return opened;
    if (auto opened = Core::check(partials.open(device, (histogramSize + Shaders::splatScanTile - 1) / Shaders::splatScanTile)); !opened)
        return opened;
    if (auto opened = Core::check(splats.open(device, std::size_t{count} * 2)); !opened)
        return opened;
    capacity = count;
    return {};
}

std::expected<void, Core::Error> GaussianSplats::open(NoorRHI::Device& device) {
    const NoorRHI::GraphicsPipelineDesc desc{
        .vertex = *Shaders::splatVertex[0],
        .fragment = *Shaders::splatFragment[0],
        .state = {.cull = NoorRHI::CullMode::None,
                  .depthCompare = NoorRHI::CompareOp::LessOrEqual,
                  .depthTest = true,
                  .depthWrite = false,
                  .blend = NoorRHI::BlendMode::PremultipliedAlpha},
        .colorFormat = format,
        .depthFormat = NoorRHI::ImageFormat::D32Float,
    };
    if (auto opened = Core::check(pipeline.open(device, desc)); !opened)
        return opened;
    if (auto opened = Core::check(cameraBuffer.open(device, 1)); !opened)
        return opened;
    if (auto opened = Core::check(state.open(device, Shaders::splatStateCount)); !opened)
        return opened;
    if (auto opened = Core::check(dispatches.open(device, Shaders::splatDispatchCount)); !opened)
        return opened;
    if (auto opened = Core::check(draw.open(device, 1)); !opened)
        return opened;
    if (auto opened = Core::check(counters.open(device, Shaders::splatCounterCount)); !opened)
        return opened;
    return Core::check(quadIndices.open(device, quadCorners.size()));
}

std::expected<void, Core::Error> GaussianSplats::sort(NoorRHI::Device& device, NoorRHI::CommandList& list, PipelineCache& pipelines,
                                                      Shaders::SplatArguments arguments) {
    auto project = getPipeline(device, pipelines, Shaders::projectSplats[0]);
    if (!project)
        return std::unexpected(std::move(project.error()));
    auto prepare = getPipeline(device, pipelines, Shaders::prepareSplatSort[0]);
    if (!prepare)
        return std::unexpected(std::move(prepare.error()));
    auto countDigits = getPipeline(device, pipelines, Shaders::countSplatDigits[0]);
    if (!countDigits)
        return std::unexpected(std::move(countDigits.error()));
    auto reduce = getPipeline(device, pipelines, Shaders::reduceSplatHistogram[0]);
    if (!reduce)
        return std::unexpected(std::move(reduce.error()));
    auto scanPartials = getPipeline(device, pipelines, Shaders::scanSplatPartials[0]);
    if (!scanPartials)
        return std::unexpected(std::move(scanPartials.error()));
    auto downsweep = getPipeline(device, pipelines, Shaders::downsweepSplatHistogram[0]);
    if (!downsweep)
        return std::unexpected(std::move(downsweep.error()));
    auto scatterDigits = getPipeline(device, pipelines, Shaders::scatterSplatDigits[0]);
    if (!scatterDigits)
        return std::unexpected(std::move(scatterDigits.error()));
    const NoorRHI::Ptr<NoorRHI::DispatchSize> blockGroups = dispatches.getPtr() + Shaders::splatDispatchBlocks;
    const NoorRHI::Ptr<NoorRHI::DispatchSize> scanGroups = dispatches.getPtr() + Shaders::splatDispatchScan;
    list.dispatch(**project, getBlockDispatch(getBlockCount(arguments.count)), arguments);
    list.dispatch(**prepare, NoorRHI::DispatchSize{1, 1, 1}, arguments);
    //! An even number of passes leaves the sorted pairs in keys and values.
    for (std::uint32_t shift = 0; shift < Shaders::splatKeyBits; shift += Shaders::splatDigitBits) {
        arguments.shift = shift;
        list.dispatchIndirect(**countDigits, blockGroups, arguments);
        list.dispatchIndirect(**reduce, scanGroups, arguments);
        list.dispatch(**scanPartials, NoorRHI::DispatchSize{Shaders::splatGroupSize, 1, 1}, arguments);
        list.dispatchIndirect(**downsweep, scanGroups, arguments);
        list.dispatchIndirect(**scatterDigits, blockGroups, arguments);
        std::swap(arguments.keys, arguments.sortedKeys);
        std::swap(arguments.values, arguments.sortedValues);
    }
    return {};
}

std::expected<void, Core::Error> GaussianSplats::record(NoorRHI::Device& device, NoorRHI::CommandList& list, PipelineCache& pipelines,
                                                        const Gpu::GpuScene& scene, const Shaders::CameraRecord& camera, const NoorRHI::Image& depth,
                                                        const NoorRHI::float2 jitter) {
    if (!image)
        return Core::fail(Core::ErrorCode::InvalidState, "the splat image has no size");
    const std::uint32_t count = scene.getGaussianCount();
    if (count == 0)
        return Core::fail(Core::ErrorCode::InvalidState, "the scene has no Gaussians to splat");
    if (auto reserved = reserve(device, count); !reserved)
        return reserved;
    const bool opening = !pipeline;
    if (opening)
        if (auto opened = open(device); !opened)
            return opened;
    const Shaders::SplatArguments arguments{
        .scene = scene.getRecord(),
        .camera = cameraBuffer.getPtr(),
        .keys = keys.getPtr(),
        .values = values.getPtr(),
        .sortedKeys = sortedKeys.getPtr(),
        .sortedValues = sortedValues.getPtr(),
        .histogram = histogram.getPtr(),
        .partials = partials.getPtr(),
        .splats = splats.getPtr(),
        .state = state.getPtr(),
        .dispatches = dispatches.getPtr(),
        .draw = draw.getPtr(),
        .counters = counters.getPtr(),
        .jitter = jitter,
        .count = count,
        .cloudCount = scene.getHostRecord().counts.z,
    };
    list.pushLabel("GaussianSplats");
    if (opening) {
        list.upload(quadIndices, std::span<const std::uint32_t>(quadCorners));
        auto reset = getPipeline(device, pipelines, Shaders::resetSplats[0]);
        if (!reset)
            return std::unexpected(std::move(reset.error()));
        list.dispatch(**reset, NoorRHI::DispatchSize{1, 1, 1}, arguments);
    }
    if (!sorted || scene.getLastChange() != Scene::Change::None || !sameView(camera, sortedCamera)) {
        list.upload(cameraBuffer, std::span<const Shaders::CameraRecord>(&camera, 1));
        if (auto recorded = sort(device, list, pipelines, arguments); !recorded)
            return recorded;
        sorted = true;
        sortedCamera = camera;
    }
    list.beginRender({.color = &image, .clear = true, .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}});
    list.endRender();
    list.beginRender({.color = &image, .depth = &depth, .clear = false});
    list.drawIndirect(pipeline, quadIndices.getPtr(), draw.getPtr(), 1, arguments);
    list.endRender();
    list.popLabel();
    return {};
}

}
