#include "Viewport.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>

#include "Logging/Log.h"
#include "Realtime/ShaderLoading.h"
#include "Scene/Billboard.h"
#include "Scene/Scene.h"
#include "Scene/SceneObject.h"

namespace
{

constexpr uint32_t ViewportGroupSize = 16;
// Scene coordinates are centimetres. Icons fade in from 25 to 50 cm, remain
// fully visible to 20 m, then fade out over the next 5 m.
constexpr glm::vec4 ViewportBillboardDistanceFade{25.0f, 50.0f, 2000.0f, 2500.0f};
constexpr float ViewportVolumeLineWidth = 1.0f;
constexpr float ViewportVolumeOpacity = 0.9f;

}

Viewport::Viewport(noorrhi::Device& gpu_device, const uint32_t width, const uint32_t height,
                   const uint32_t traceWidth, const uint32_t traceHeight,
                   const uint32_t imageWidth, const uint32_t imageHeight,
                   const ViewportInputs& inputs, const noorrhi::ImageFormat outputImageFormat,
                   const bool exportOutputMemory, const bool externalOutput)
: gpuDevice(gpu_device), logicalWidth(width), logicalHeight(height)
, traceWidth_(traceWidth), traceHeight_(traceHeight)
, exportOutputMemory_(exportOutputMemory), externalOutput_(externalOutput), inputs(inputs)
{
    createOutputResources(imageWidth, imageHeight, outputImageFormat);
    shader = loadShader(gpuDevice, "Viewport/Viewport.spv");
    pipeline = gpuDevice.compute(shader);
    createBillboardPipeline();
    createBillboardTextures();
    createVolumePipeline();
    {
        lightIdClearShader = loadShader(gpuDevice, "Viewport/ViewportBillboards.spv", "lightIdClear");
        lightIdStampShader = loadShader(gpuDevice, "Viewport/ViewportBillboards.spv", "lightIdStamp");
        lightIdClearPipeline = gpuDevice.compute(lightIdClearShader);
        lightIdStampPipeline = gpuDevice.compute(lightIdStampShader);
    }
    reserveBillboards(1);
}

Viewport::~Viewport()
{
    NR_LOG_INFO("Destroying Viewport");
}

void Viewport::createOutputResources(const uint32_t width, const uint32_t height,
                                 const noorrhi::ImageFormat format)
{
    if (!externalOutput_)
        outputImage = gpuDevice.image<std::byte>(width, height,
            noorrhi::ImageUsage::Sampled | noorrhi::ImageUsage::Storage
                | noorrhi::ImageUsage::ColorAttachment
                | (exportOutputMemory_ ? noorrhi::ImageUsage::ExternalMemory
                                       : noorrhi::ImageUsage{}),
            format);
    outputFormat_ = format;
    outputImageWidth_ = width;
    outputImageHeight_ = height;
    // Replaced with the output, and with nothing worth carrying over: the
    // field is indexed in output pixels.
    selectionSdfImage = gpuDevice.image<std::byte>(width, height,
        noorrhi::ImageUsage::Sampled | noorrhi::ImageUsage::Storage,
        noorrhi::ImageFormat::R32Float);
    outlineSampleCount_ = 0;
    // Indexed like the output image: row stride is its allocated width.
    lightIdBuffer = gpuDevice.buffer<std::uint32_t>(
        static_cast<std::size_t>(width) * height);
    volumeIdBuffer = gpuDevice.buffer<std::uint32_t>(
        static_cast<std::size_t>(width) * height);
    lightIdsClear_ = false;
    volumeIdsClear_ = false;
}

void Viewport::createBillboardPipeline()
{
    billboardVertexShader = loadShader(gpuDevice, "Viewport/ViewportBillboards.spv", "vertMain");
    billboardFragmentShader = loadShader(gpuDevice, "Viewport/ViewportBillboards.spv", "fragMain");
    noorrhi::GraphicsState state{};
    state.cull = noorrhi::CullMode::None;
    state.depth_test = false;
    state.depth_write = false;
    state.blend.enabled = true;
    billboardPipeline = gpuDevice.graphics({billboardVertexShader,
        billboardFragmentShader, state, outputFormat_});

}

void Viewport::createBillboardTextures()
{
    const uint32_t size = static_cast<uint32_t>(nr::graphics::ViewportBillboardTextureSize);
    billboardSampler = gpuDevice.sampler({noorrhi::Filter::Linear, noorrhi::AddressMode::ClampToEdge,
        noorrhi::AddressMode::ClampToEdge, noorrhi::AddressMode::ClampToEdge});
    std::vector<std::uint32_t> handles;
    for (const std::string_view name : BillboardIconNames)
    {
        const std::string path = "billboards/" + std::string(name) + ".msdf";
        const std::span<const std::byte> texels = embeddedFile(path);
        if (texels.size() != static_cast<std::size_t>(size) * size * 4)
            throw std::runtime_error("billboard texture has the wrong size: " + path);
        // Distances, not colors: no sRGB decoding.
        noorrhi::Image<std::byte>& texture = billboardTextures.emplace_back(gpuDevice.image<std::byte>(
            size, size, noorrhi::ImageUsage::Sampled, noorrhi::ImageFormat::Rgba8Unorm));
        texture.upload(texels);
        handles.push_back(texture.sampled_handle().value);
    }
    billboardTextureHandles = gpuDevice.buffer<std::uint32_t>(handles.size());
    billboardTextureHandles.upload(std::span<const std::uint32_t>(handles));
}

void Viewport::createVolumePipeline()
{
    volumeVertexShader = loadShader(gpuDevice, "Viewport/ViewportVolumes.spv", "vertMain");
    volumeFragmentShader = loadShader(gpuDevice, "Viewport/ViewportVolumes.spv", "fragMain");
    noorrhi::GraphicsState state{};
    state.cull = noorrhi::CullMode::None;
    state.depth_test = false;
    state.depth_write = false;
    state.blend.enabled = true;
    volumePipeline = gpuDevice.graphics({volumeVertexShader,
        volumeFragmentShader, state, outputFormat_});
}

bool Viewport::reserveBillboards(const uint32_t capacity)
{
    if (capacity <= billboardCapacity)
        return false;

    // The buffer may still be referenced by an in-flight command buffer from a
    // previous frame; growth is rare (only when the light count exceeds the
    // current capacity), so waiting here is cheap insurance.
    if (billboardCapacity > 0)
        gpuDevice.synchronize();
    billboardBuffer = gpuDevice.buffer<std::byte>(
        static_cast<std::size_t>(capacity) * sizeof(ViewportBillboard));
    billboardCapacity = capacity;
    billboardEntry = billboardBuffer.ptr();
    return true;
}

namespace
{
ViewportBillboard makeBillboard(const SceneObject& object)
{
    const Billboard& billboard = *object.getBillboard();
    return ViewportBillboard{
        glm::vec4(object.getWorldTransform().getPosition(),
            static_cast<float>(billboard.icon)),
        glm::vec4(billboard.color, object.isVisible() ? 1.0f : 0.0f)};
}
}

void Viewport::updateOverlays(Scene& scene)
{
    updateBillboards(scene);
    updateVolumes(scene);
}

void Viewport::updateBillboards(Scene& scene)
{
    const std::vector<SceneObject*>& objects = scene.getBillboardObjects();
    std::vector<uint32_t> changed = scene.takeChangedBillboardSlots();
    if (changed.empty() && objects.size() == billboardData.size())
        return;

    billboardData.resize(objects.size());
    billboardHandles.resize(objects.size());
    for (const uint32_t slot : changed)
    {
        billboardData[slot] = makeBillboard(*objects[slot]);
        billboardHandles[slot] = objects[slot]->getHandle();
    }
    billboardCount = static_cast<uint32_t>(objects.size());

    // A replaced buffer is empty, so it takes every record again.
    if (billboardCount > billboardCapacity
        && reserveBillboards(std::max(billboardCount, billboardCapacity * 2)))
    {
        billboardBuffer.upload(std::as_bytes(
            std::span<const ViewportBillboard>(billboardData.data(), billboardCount)));
        return;
    }

    std::ranges::sort(changed);
    // One upload per run of neighbours; past a few dozen runs a single
    // covering upload is cheaper than that many submissions.
    constexpr std::size_t MaxRuns = 32;
    std::vector<std::pair<uint32_t, uint32_t>> runs;
    for (const uint32_t index : changed)
    {
        if (!runs.empty() && index == runs.back().second + 1)
            runs.back().second = index;
        else
            runs.emplace_back(index, index);
    }
    if (runs.size() > MaxRuns)
        runs = {{runs.front().first, runs.back().second}};
    for (const auto [first, last] : runs)
        billboardBuffer.upload(std::as_bytes(std::span<const ViewportBillboard>(
            billboardData.data() + first, last - first + 1)), first * sizeof(ViewportBillboard));
}

void Viewport::updateVolumes(Scene& scene)
{
    const auto& objects = scene.getVolumeObjects();
    const std::vector<uint32_t> changed = scene.takeChangedVolumeSlots();
    bool rebuild = observedVolumeStructureRevision != scene.getVolumeStructureRevision()
        || volumeSlotToDraw.size() != objects.size();
    if (!rebuild)
        for (const uint32_t slot : changed) {
            const uint32_t draw = volumeSlotToDraw[slot];
            const bool drawable = objects[slot]->isVisible()
                && objects[slot]->getOutline()->segmentCount() > 0;
            if (drawable != (draw != ~0u)) {
                rebuild = true;
                break;
            }
        }
    if (!rebuild && changed.empty())
        return;
    if (rebuild) {
        volumeDraws.clear();
        volumeSlotToDraw.assign(objects.size(), ~0u);
        for (uint32_t slot = 0; slot < objects.size(); ++slot) {
            const VolumeInstance& volume = *objects[slot];
            if (!volume.isVisible() || volume.getOutline()->segmentCount() == 0)
                continue;
            const noorrhi::Buffer<glm::vec3>& edges = volume.getOutline()->upload(gpuDevice);
            const uint32_t index = static_cast<uint32_t>(volumeDraws.size());
            volumeSlotToDraw[slot] = index;
            volumeDraws.push_back({volume.getHandle(), volume.getOutline(),
                volume.getWorldTransform().getMatrix(),
                glm::vec4(volume.getColor(), ViewportVolumeOpacity), edges.ptr()});
        }
        observedVolumeStructureRevision = scene.getVolumeStructureRevision();
        return;
    }
    // Movement changes only the draw's transform; the shared edge buffer
    // needs no upload.
    for (const uint32_t slot : changed) {
        const uint32_t draw = volumeSlotToDraw[slot];
        if (draw == ~0u)
            continue;
        volumeDraws[draw].world = objects[slot]->getWorldTransform().getMatrix();
    }
}

std::optional<uint32_t> Viewport::billboardAt(const uint32_t x, const uint32_t y) const
{
    if (!lightIdBuffer || x >= logicalWidth || y >= logicalHeight)
        return std::nullopt;
    std::uint32_t value = 0;
    lightIdBuffer.download(std::span<std::uint32_t>(&value, 1),
        static_cast<std::size_t>(y) * outputImageWidth_ + x);
    if (value == 0 || value > billboardData.size())
        return std::nullopt;
    return value - 1;
}

SceneObjectHandle Viewport::billboardObjectAt(const uint32_t x, const uint32_t y) const
{
    const auto index = billboardAt(x, y);
    return index && *index < billboardHandles.size() ? billboardHandles[*index] : SceneObjectHandle{};
}

SceneObjectHandle Viewport::volumeObjectAt(const uint32_t x, const uint32_t y) const
{
    if (!volumeIdBuffer || x >= logicalWidth || y >= logicalHeight)
        return {};
    std::uint32_t value = 0;
    volumeIdBuffer.download(std::span<std::uint32_t>(&value, 1),
        static_cast<std::size_t>(y) * outputImageWidth_ + x);
    return value > 0 && value <= volumeDraws.size()
        ? volumeDraws[value - 1].handle : SceneObjectHandle{};
}

std::optional<glm::vec3> Viewport::billboardPositionAt(const uint32_t x, const uint32_t y) const
{
    const auto index = billboardAt(x, y);
    if (!index)
        return std::nullopt;
    return glm::vec3(billboardData[*index].positionType);
}

void Viewport::drawBillboards(const glm::mat4& viewProjection,
    const SceneObjectHandle selectedObject, const ViewportOutput& output)
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
        glm::vec2(static_cast<float>(logicalWidth) / static_cast<float>(output.width),
                  static_cast<float>(logicalHeight) / static_cast<float>(output.height)),
        ViewportBillboardHalfSize,
        billboardCount,
        lightIdBuffer.ptr().address,
        output.width,
        selectedBillboard,
        glm::vec4(billboardCameraPosition_, hasBillboardCamera_ ? 1.0f : 0.0f),
        ViewportBillboardDistanceFade,
        billboardTextureHandles.ptr().address,
        billboardSampler.handle().value};
    gpuDevice.render({.color = output.image, .clear = false, .flip_y = false},
        [this, &arguments] {
            billboardPipeline.draw_instanced(6, billboardCount, arguments);
        });

    lightIdStampPipeline.launch({billboardCount, 1, 1}, arguments);
}

void Viewport::drawVolumes(const glm::mat4& viewProjection,
    const SceneObjectHandle selectedObject, const ViewportOutput& output)
{
    gpuDevice.render({.color = output.image, .clear = false, .flip_y = false},
        [&] {
            for (uint32_t index = 0; index < volumeDraws.size(); ++index)
            {
                const VolumeDraw& volume = volumeDraws[index];
                const nr::graphics::ViewportVolumePushConstants arguments{
                    glm::transpose(viewProjection * volume.world),
                    volume.segments.address,
                    glm::vec2(static_cast<float>(logicalWidth),
                              static_cast<float>(logicalHeight)),
                    glm::vec2(static_cast<float>(logicalWidth) / static_cast<float>(output.width),
                              static_cast<float>(logicalHeight) / static_cast<float>(output.height)),
                    volume.handle == selectedObject
                        ? glm::vec4(nr::graphics::ViewportSelectionColor, 1.0f) : volume.color,
                    ViewportVolumeLineWidth,
                    inputs.depth.value,
                    glm::vec2(static_cast<float>(traceWidth_), static_cast<float>(traceHeight_)),
                    inputs.depthJitter,
                    volumeIdBuffer.ptr().address,
                    output.width,
                    index};
                volumePipeline.draw_instanced(6, volume.outline->segmentCount(), arguments);
            }
        });
}

void Viewport::dispatch(
    const uint32_t selectedCryptomatteId,
    const bool restartOutline,
    const glm::mat4& viewProjection,
    const float exposure,
    const int bufferVisualization,
    const bool tonemappingEnabled,
    const bool showBillboards,
    const SceneObjectHandle selectedObject, const ViewportOutput suppliedOutput)
{
    // Before the first resize the inputs do not exist yet; skip until a later
    // call supplies them.
    if (!inputs)
        return;
    const ViewportOutput output = suppliedOutput ? suppliedOutput : ViewportOutput{
        outputImage.handle(), outputImage.storage_handle(), outputImageWidth_, outputImageHeight_, outputFormat_};
    if (!output || output.width != outputImageWidth_ || output.height != outputImageHeight_
        || output.format != outputFormat_)
        throw std::invalid_argument("Viewport output does not match its allocated presentation resources");
    // A view of an image the renderer does not write takes the shader's
    // invalid-view case, which draws black.
    const noorrhi::TextureHandle viewed[] = {inputs.color, inputs.albedo, inputs.normal,
        inputs.crypto, inputs.position};
    const bool unavailable = bufferVisualization >= 0
        && bufferVisualization < static_cast<int>(std::size(viewed)) && !viewed[bufferVisualization];
    const int shownVisualization = unavailable ? -1 : bufferVisualization;

    // The distance field is screen-space, so it restarts when the camera or
    // scene moves what it measures, when selection changes, or on layout
    // changes (see resize()). Its history is independent of beauty accumulation.
    if (restartOutline || selectedCryptomatteId != outlinedCryptomatteId_)
        outlineSampleCount_ = 0;
    outlinedCryptomatteId_ = selectedCryptomatteId;

    const nr::graphics::ViewportCompositePushConstants arguments{
        inputs.color.value,
        output.storage.value,
        inputs.crypto.value,
        inputs.albedo.value,
        inputs.normal.value,
        inputs.position.value,
        selectedCryptomatteId, exposure, shownVisualization, tonemappingEnabled ? 1 : 0,
        logicalWidth, logicalHeight,
        // Both are the logical size until a renderer reports otherwise, which
        // leaves the ratio at 1 for every non-upscaling path.
        traceWidth_ > 0 ? static_cast<float>(logicalWidth) / static_cast<float>(traceWidth_)
                        : 1.0f,
        selectionSdfImage.storage_handle().value,
        outlineSampleCount_,
        static_cast<float>(traceWidth_), static_cast<float>(traceHeight_),
        inputs.depthJitter.x, inputs.depthJitter.y};
    if (selectedCryptomatteId != ~0u)
        ++outlineSampleCount_;
    // One texel past the logical edge carries a copy of the edge; see the shader.
    const uint32_t coveredWidth = std::min(logicalWidth + 1u, output.width);
    const uint32_t coveredHeight = std::min(logicalHeight + 1u, output.height);
    const uint32_t groupCountX =
        (coveredWidth + ViewportGroupSize - 1) / ViewportGroupSize;
    const uint32_t groupCountY =
        (coveredHeight + ViewportGroupSize - 1) / ViewportGroupSize;
    gpuDevice.label("Viewport Composite", [&] {
        pipeline.launch({groupCountX, groupCountY, 1}, arguments);
    });

    // The id buffers are reset every frame something draws into them, so hidden
    // icons are never pickable and nothing stale survives a camera move. Once
    // clear they stay so until something draws again.
    const bool drawsVolumes = showBillboards && !volumeDraws.empty();
    const bool drawsBillboards = showBillboards && billboardCount > 0;
    nr::graphics::ViewportBillboardPushConstants clearArguments{};
    clearArguments.screenSize = glm::vec2(static_cast<float>(logicalWidth),
        static_cast<float>(logicalHeight));
    clearArguments.lightIdStride = output.width;
    const auto clearIds = [&](const char* name, const noorrhi::Buffer<std::uint32_t>& ids,
                              const bool draws, bool& clear) {
        if (!draws && clear)
            return;
        clearArguments.lightIds = ids.ptr().address;
        gpuDevice.label(name, [&] {
            lightIdClearPipeline.launch({(logicalWidth + ViewportGroupSize - 1) / ViewportGroupSize,
                (logicalHeight + ViewportGroupSize - 1) / ViewportGroupSize, 1}, clearArguments);
        });
        clear = !draws;
    };
    clearIds("Viewport Light ID Clear", lightIdBuffer, drawsBillboards, lightIdsClear_);
    clearIds("Viewport Volume ID Clear", volumeIdBuffer, drawsVolumes, volumeIdsClear_);
    gpuDevice.barrier(noorrhi::Stage::Compute, noorrhi::Stage::Fragment);

    if (drawsVolumes)
        gpuDevice.label("Viewport Volume Outlines", [&] {
            drawVolumes(viewProjection, selectedObject, output);
        });
    if (drawsBillboards)
        gpuDevice.label("Viewport Billboards", [&] {
            drawBillboards(viewProjection, selectedObject, output);
        });
    // Picking reads the buffer back with a copy after this frame.
    gpuDevice.barrier(noorrhi::Stage::Compute, noorrhi::Stage::Copy);
    gpuDevice.barrier(noorrhi::Stage::Fragment, noorrhi::Stage::Copy);
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
    const bool outputChanged = outputImageWidth_ != imageWidth
        || outputImageHeight_ != imageHeight
        || outputFormat_ != outputImageFormat;
    if (outputChanged)
    {
        // Only replacing the viewport resources requires waiting for work
        // that may still reference the old image or format-specific pipeline.
        gpuDevice.synchronize();
        const noorrhi::ImageFormat previousFormat = outputFormat_;
        createOutputResources(imageWidth, imageHeight, outputImageFormat);
        // The billboard pipeline bakes in its color-attachment format, so it
        // only has to be rebuilt when that format actually changes.
        if (outputImageFormat != previousFormat)
        {
            createBillboardPipeline();
            createVolumePipeline();
        }
    }
    inputs = newInputs;
}

std::vector<noorrhi::float4> Viewport::readOutput() const
{
    if (externalOutput_)
        throw noorrhi::Error(noorrhi::ErrorCode::InvalidArgument,
            "Viewport::readOutput is unavailable with an external output target");
    if (outputFormat_ != noorrhi::ImageFormat::Rgba32Float)
        throw noorrhi::Error(noorrhi::ErrorCode::InvalidArgument,
            "Viewport::readOutput requires an RGBA32F output texture");
    const uint32_t imageWidth = outputImageWidth_;
    std::vector<noorrhi::float4> image(static_cast<std::size_t>(imageWidth)
        * outputImageHeight_);
    outputImage.download(std::as_writable_bytes(std::span(image)));
    if (imageWidth == logicalWidth && outputImageHeight_ == logicalHeight)
        return image;
    std::vector<noorrhi::float4> result(static_cast<std::size_t>(logicalWidth)
        * logicalHeight);
    for (uint32_t y = 0; y < logicalHeight; ++y)
        std::copy_n(image.begin() + static_cast<std::ptrdiff_t>(y) * imageWidth,
            logicalWidth, result.begin() + static_cast<std::ptrdiff_t>(y) * logicalWidth);
    return result;
}
