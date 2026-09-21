#include "Viewport.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <span>

#include "Logging/Log.h"
#include "Scene/Scene.h"
#include "Scene/LightInstance.h"

namespace
{
alignas(uint32_t) constexpr unsigned char noorRayViewportSpv[] = {
    #embed "Viewport/Viewport.spv"
};
constexpr std::size_t noorRayViewportSpvLength = sizeof(noorRayViewportSpv);

alignas(uint32_t) constexpr unsigned char noorRayViewportBillboardsSpv[] = {
    #embed "Viewport/ViewportBillboards.spv"
};
constexpr std::size_t noorRayViewportBillboardsSpvLength = sizeof(noorRayViewportBillboardsSpv);

constexpr uint32_t ViewportGroupSize = 16;

std::span<const std::byte> shader_bytes(const unsigned char* data, const std::size_t size)
{
    return {reinterpret_cast<const std::byte*>(data), size};
}

}

Viewport::Viewport(noorrhi::Device& gpu_device, const uint32_t width, const uint32_t height,
                   const uint32_t traceWidth, const uint32_t traceHeight,
                   const uint32_t imageWidth, const uint32_t imageHeight,
                   const ViewportInputs& inputs, const noorrhi::ImageFormat outputImageFormat,
                   const bool exportOutputMemory)
: gpuDevice(gpu_device), logicalWidth(width), logicalHeight(height)
, traceWidth_(traceWidth), traceHeight_(traceHeight)
, exportOutputMemory_(exportOutputMemory), inputs(inputs)
{
    createOutputImage(imageWidth, imageHeight, outputImageFormat);
    shader = gpuDevice.create_shader(
        shader_bytes(noorRayViewportSpv, noorRayViewportSpvLength));
    pipeline = gpuDevice.compute(shader);
    createBillboardPipeline();
    {
        const auto bytes = shader_bytes(noorRayViewportBillboardsSpv,
            noorRayViewportBillboardsSpvLength);
        lightIdClearShader = gpuDevice.create_shader(bytes, "lightIdClear");
        lightIdStampShader = gpuDevice.create_shader(bytes, "lightIdStamp");
        lightIdClearPipeline = gpuDevice.compute(lightIdClearShader);
        lightIdStampPipeline = gpuDevice.compute(lightIdStampShader);
    }
    reserveBillboards(1);
}

Viewport::~Viewport()
{
    NR_LOG_INFO("Destroying Viewport");
}

void Viewport::createOutputImage(const uint32_t width, const uint32_t height,
                                 const noorrhi::ImageFormat format)
{
    outputImage = gpuDevice.image<std::byte>(width, height,
        noorrhi::ImageUsage::Sampled | noorrhi::ImageUsage::Storage
            | noorrhi::ImageUsage::ColorAttachment
            | (exportOutputMemory_ ? noorrhi::ImageUsage::ExternalMemory
                                   : noorrhi::ImageUsage{}),
        format);
    outputFormat_ = format;
    // Replaced with the output, and with nothing worth carrying over: the
    // field is indexed in output pixels.
    selectionSdfImage = gpuDevice.image<std::byte>(width, height,
        noorrhi::ImageUsage::Sampled | noorrhi::ImageUsage::Storage,
        noorrhi::ImageFormat::R32Float);
    outlineSampleCount_ = 0;
    // Indexed like the output image: row stride is its allocated width.
    lightIdBuffer = gpuDevice.buffer<std::uint32_t>(
        static_cast<std::size_t>(width) * height);
}

void Viewport::createBillboardPipeline()
{
    const auto bytes = shader_bytes(noorRayViewportBillboardsSpv,
        noorRayViewportBillboardsSpvLength);
    billboardVertexShader = gpuDevice.create_shader(bytes, "vertMain");
    billboardFragmentShader = gpuDevice.create_shader(bytes, "fragMain");
    noorrhi::GraphicsState state{};
    state.cull = noorrhi::CullMode::None;
    state.depth_test = false;
    state.depth_write = false;
    state.blend.enabled = true;
    billboardPipeline = gpuDevice.graphics({billboardVertexShader,
        billboardFragmentShader, state, outputFormat_});

}

void Viewport::reserveBillboards(const uint32_t capacity)
{
    if (capacity <= billboardCapacity)
        return;

    // The buffer may still be referenced by an in-flight command buffer from a
    // previous frame; growth is rare (only when the light count exceeds the
    // current capacity), so waiting here is cheap insurance.
    if (billboardCapacity > 0)
        gpuDevice.synchronize();
    billboardBuffer = gpuDevice.buffer<std::byte>(
        static_cast<std::size_t>(capacity) * sizeof(ViewportBillboard));
    billboardCapacity = capacity;
    billboardEntry = billboardBuffer.ptr();
}

namespace
{
ViewportBillboard makeBillboard(const LightInstance& light)
{
    return ViewportBillboard{
        glm::vec4(light.getWorldTransform().getPosition(),
            static_cast<float>(light.getLightType())),
        glm::vec4(light.getColor(), 1.0f)};
}
}

void Viewport::updateBillboards(const Scene& scene)
{
    if (observedLightRevision == scene.getLightRevision()
        && observedHierarchyRevision == scene.getHierarchyRevision())
        return;

    const uint32_t lightCount = scene.getPointLightCount()
        + scene.getSpotLightCount()
        + scene.getRectLightCount()
        + scene.getDirectionalLightCount();
    bool rebuild = observedHierarchyRevision != scene.getHierarchyRevision()
        || billboardHandles.size() != lightCount;
    if (!rebuild)
    {
        // A light moved or was edited: the set is unchanged, so revisit only
        // the known lights and upload the span of records that changed.
        std::size_t first = billboardData.size();
        std::size_t last = 0;
        for (std::size_t index = 0; index < billboardHandles.size(); ++index)
        {
            const auto* light = dynamic_cast<const LightInstance*>(
                scene.getObject(billboardHandles[index]));
            if (!light)
            {
                rebuild = true;
                break;
            }
            const ViewportBillboard record = makeBillboard(*light);
            if (std::memcmp(&record, &billboardData[index], sizeof(record)) == 0)
                continue;
            billboardData[index] = record;
            first = std::min(first, index);
            last = index;
        }
        if (!rebuild && first < billboardData.size())
            billboardBuffer.upload(std::as_bytes(std::span<const ViewportBillboard>(
                billboardData.data() + first, last - first + 1)),
                first * sizeof(ViewportBillboard));
    }
    if (rebuild)
        rebuildBillboards(scene);
    observedLightRevision = scene.getLightRevision();
    observedHierarchyRevision = scene.getHierarchyRevision();
}

void Viewport::rebuildBillboards(const Scene& scene)
{
    billboardData.clear();
    billboardHandles.clear();
    for (const auto& obj : scene.getSceneObjects())
    {
        if (const auto* light = dynamic_cast<const LightInstance*>(obj.get()))
        {
            billboardData.push_back(makeBillboard(*light));
            billboardHandles.push_back(light->getHandle());
        }
    }
    billboardCount = static_cast<uint32_t>(billboardData.size());
    reserveBillboards(std::max(1u, billboardCount));
    if (billboardCount > 0)
        billboardBuffer.upload(std::as_bytes(
            std::span<const ViewportBillboard>(billboardData.data(), billboardCount)));
}

std::optional<uint32_t> Viewport::billboardAt(const uint32_t x, const uint32_t y) const
{
    if (!lightIdBuffer || x >= logicalWidth || y >= logicalHeight)
        return std::nullopt;
    std::uint32_t value = 0;
    lightIdBuffer.download(std::span<std::uint32_t>(&value, 1),
        static_cast<std::size_t>(y) * outputImage.width() + x);
    if (value == 0 || value > billboardData.size())
        return std::nullopt;
    return value - 1;
}

SceneObjectHandle Viewport::lightAt(const uint32_t x, const uint32_t y) const
{
    const auto index = billboardAt(x, y);
    return index && *index < billboardHandles.size() ? billboardHandles[*index] : SceneObjectHandle{};
}

std::optional<glm::vec3> Viewport::lightPositionAt(const uint32_t x, const uint32_t y) const
{
    const auto index = billboardAt(x, y);
    if (!index)
        return std::nullopt;
    return glm::vec3(billboardData[*index].positionType);
}

void Viewport::drawBillboards(const glm::mat4& viewProjection,
    const SceneObjectHandle selectedObject)
{
    uint32_t selectedBillboard = ~0u;
    if (selectedObject.isValid())
        if (const auto found = std::ranges::find(billboardHandles, selectedObject);
            found != billboardHandles.end())
            selectedBillboard = static_cast<uint32_t>(found - billboardHandles.begin());

    // The path-traced image uses bottom-left row order and is flipped once by
    // ImGui. Draw overlays into that same raw orientation so their projected
    // position receives the identical presentation flip. clear=false keeps the
    // composite pass's output that this draws on top of.
    const nr::graphics::ViewportBillboardPushConstants arguments{
        // Slang emits this root-argument matrix with row-major storage. GLM
        // stores matrices column-major, so transpose once at the ABI boundary
        // to preserve the same mathematical matrix in the shader.
        glm::transpose(viewProjection),
        billboardEntry.address,
        glm::vec2(static_cast<float>(logicalWidth),
                  static_cast<float>(logicalHeight)),
        glm::vec2(static_cast<float>(logicalWidth) / static_cast<float>(outputImage.width()),
                  static_cast<float>(logicalHeight) / static_cast<float>(outputImage.height())),
        ViewportBillboardMinPixelRadius,
        ViewportBillboardMaxPixelRadius,
        ViewportBillboardNearDistance,
        ViewportBillboardFarDistance,
        1.0f,
        billboardCount,
        lightIdBuffer.ptr().address,
        outputImage.width(),
        selectedBillboard};
    gpuDevice.render({.color = outputImage.handle(), .clear = false, .flip_y = false},
        [this, &arguments] {
            billboardPipeline.draw_instanced(6, billboardCount, arguments);
        });

    auto pickArguments = arguments;
    pickArguments.radiusScale = ViewportBillboardPickScale;
    lightIdStampPipeline.launch({billboardCount, 1, 1}, pickArguments);
}

void Viewport::dispatch(
    const uint32_t selectedCryptomatteId,
    const bool restartOutline,
    const glm::mat4& viewProjection,
    const float exposure,
    const int bufferVisualization,
    const int gaussianOverdrawMax,
    const bool tonemappingEnabled,
    const bool showBillboards,
    const SceneObjectHandle selectedObject)
{
    // Before the first resize, or with AOVs switched off, some inputs do not
    // exist yet; skip until a later call supplies the full complement.
    if (!inputs || !outputImage)
        return;

    // The distance field is screen-space, so it restarts with everything that
    // moves what it measures: the renderer's accumulation (camera or scene),
    // the selection, or the layout (see resize()).
    if (restartOutline || selectedCryptomatteId != outlinedCryptomatteId_)
        outlineSampleCount_ = 0;
    outlinedCryptomatteId_ = selectedCryptomatteId;

    const nr::graphics::ViewportCompositePushConstants arguments{
        inputs.color.value,
        outputImage.storage_handle().value,
        inputs.crypto.value,
        inputs.albedo.value,
        inputs.normal.value,
        inputs.position.value,
        inputs.overdraw.address,
        selectedCryptomatteId, exposure, bufferVisualization, tonemappingEnabled ? 1 : 0,
        static_cast<uint32_t>(std::max(gaussianOverdrawMax, 1)),
        logicalWidth, logicalHeight,
        // Both are the logical size until a renderer reports otherwise, which
        // leaves the ratio at 1 for every non-upscaling path.
        traceWidth_ > 0 ? static_cast<float>(logicalWidth) / static_cast<float>(traceWidth_)
                        : 1.0f,
        selectionSdfImage.storage_handle().value,
        outlineSampleCount_};
    if (selectedCryptomatteId != ~0u)
        ++outlineSampleCount_;
    // One texel past the logical edge carries a copy of the edge; see the shader.
    const uint32_t coveredWidth = std::min(logicalWidth + 1u, outputImage.width());
    const uint32_t coveredHeight = std::min(logicalHeight + 1u, outputImage.height());
    const uint32_t groupCountX =
        (coveredWidth + ViewportGroupSize - 1) / ViewportGroupSize;
    const uint32_t groupCountY =
        (coveredHeight + ViewportGroupSize - 1) / ViewportGroupSize;
    pipeline.launch({groupCountX, groupCountY, 1}, arguments);

    // The light-id buffer is reset every frame, so hidden icons are never
    // pickable and nothing stale survives a camera move.
    nr::graphics::ViewportBillboardPushConstants clearArguments{};
    clearArguments.screenSize = glm::vec2(static_cast<float>(logicalWidth),
        static_cast<float>(logicalHeight));
    clearArguments.lightIds = lightIdBuffer.ptr().address;
    clearArguments.lightIdStride = outputImage.width();
    lightIdClearPipeline.launch({(logicalWidth + ViewportGroupSize - 1) / ViewportGroupSize,
        (logicalHeight + ViewportGroupSize - 1) / ViewportGroupSize, 1}, clearArguments);

    if (showBillboards && billboardCount > 0)
        drawBillboards(viewProjection, selectedObject);
    // Picking reads the buffer back with a copy after this frame.
    gpuDevice.barrier(noorrhi::Stage::Compute, noorrhi::Stage::Copy);
}

void Viewport::resize(const uint32_t width, const uint32_t height,
                      const uint32_t traceWidth, const uint32_t traceHeight,
                      const uint32_t imageWidth, const uint32_t imageHeight,
                      const ViewportInputs& newInputs,
                      const noorrhi::ImageFormat outputImageFormat)
{
    if (width == 0 || height == 0 || imageWidth == 0 || imageHeight == 0)
        return;

    // The field is indexed in output pixels and scaled by the trace ratio, so
    // either one moving leaves it describing a layout that no longer exists.
    if (width != logicalWidth || height != logicalHeight
        || traceWidth != traceWidth_ || traceHeight != traceHeight_)
        outlineSampleCount_ = 0;
    logicalWidth = width;
    logicalHeight = height;
    traceWidth_ = traceWidth;
    traceHeight_ = traceHeight;
    const bool outputChanged = outputImage.width() != imageWidth
        || outputImage.height() != imageHeight
        || outputFormat_ != outputImageFormat;
    if (outputChanged)
    {
        // Only replacing the viewport resources requires waiting for work
        // that may still reference the old image or format-specific pipeline.
        gpuDevice.synchronize();
        const noorrhi::ImageFormat previousFormat = outputFormat_;
        createOutputImage(imageWidth, imageHeight, outputImageFormat);
        // The billboard pipeline bakes in its color-attachment format, so it
        // only has to be rebuilt when that format actually changes.
        if (outputImageFormat != previousFormat)
            createBillboardPipeline();
    }
    inputs = newInputs;
}

std::vector<noorrhi::float4> Viewport::readOutput() const
{
    if (outputFormat_ != noorrhi::ImageFormat::Rgba32Float)
        throw noorrhi::Error(noorrhi::ErrorCode::InvalidArgument,
            "Viewport::readOutput requires an RGBA32F output texture");
    const uint32_t imageWidth = outputImage.width();
    std::vector<noorrhi::float4> image(static_cast<std::size_t>(imageWidth)
        * outputImage.height());
    outputImage.download(std::as_writable_bytes(std::span(image)));
    if (imageWidth == logicalWidth && outputImage.height() == logicalHeight)
        return image;
    std::vector<noorrhi::float4> result(static_cast<std::size_t>(logicalWidth)
        * logicalHeight);
    for (uint32_t y = 0; y < logicalHeight; ++y)
        std::copy_n(image.begin() + static_cast<std::ptrdiff_t>(y) * imageWidth,
            logicalWidth, result.begin() + static_cast<std::ptrdiff_t>(y) * logicalWidth);
    return result;
}
