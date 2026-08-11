#include "Viewport/ViewportSession.h"

#include <algorithm>
#include <cmath>

#include "Backend/OptiX/Runtime/Raytracer.h"
#include "Backend/Vulkan/Runtime/Context.h"
#include "Backend/Vulkan/Viewport/Viewport.h"
#include "Rendering/Camera/Camera.h"
#include "Rendering/Camera/CameraInstance.h"
#include "Scene/Objects/LightInstance.h"
#include "Scene/Objects/MeshInstance.h"
#include "Scene/Scene.h"

namespace nr {

namespace {

uint32_t scaledExtent(const float value)
{
    return static_cast<uint32_t>(std::lround(std::max(1.0f, value)));
}

}  // namespace

ViewportSession::ViewportSession(
    Context& context, Scene& scene, const CreateInfo& createInfo)
    : context(context)
    , scene(scene)
    , displayFormat(createInfo.displayFormat)
    , sizing(createInfo.sizing)
    , fill(createInfo.fill)
    , sensorFit(createInfo.sensorFit)
    , targetWidth(createInfo.width)
    , targetHeight(createInfo.height)
{
    // The raytracer has to exist before the host creates any geometry:
    // MeshAsset's constructor builds its BLAS through the scene's context and
    // skips it when CUDA and OptiX are not up yet, which would leave every
    // instance pointing at a null traversable.
    raytracer = std::make_unique<Raytracer>(context, scene);
    raytracer->setAovEnabled(createInfo.aovEnabled);
    raytracer->resize(createInfo.width, createInfo.height);

    if (createInfo.applyRenderSettings)
    {
        RenderSettings& settings = scene.getRenderSettings();
        settings.samples = std::max(1, createInfo.samplesPerStep);
        settings.maxSamples = std::max(settings.samples, createInfo.maxSamples);
        settings.aovEnabled = createInfo.aovEnabled;
    }
    applySensorFit();

    compositor = std::make_unique<Viewport>(
        context, raytracer->getWidth(), raytracer->getHeight(),
        raytracer->getOutputColor(), raytracer->getOutputAlbedo(),
        raytracer->getOutputNormal(), raytracer->getOutputCrypto(),
        raytracer->getOutputPosition(), raytracer->getOutputDenoised(),
        displayFormat);

    ensureDisplayImage(raytracer->getWidth(), raytracer->getHeight());

    cryptoStagingBuffer = Buffer(context, Buffer::Type::Custom, sizeof(uint32_t),
        nullptr, vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible
            | vk::MemoryPropertyFlagBits::eHostCoherent);
    positionStagingBuffer = Buffer(context, Buffer::Type::Custom, sizeof(float) * 4,
        nullptr, vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible
            | vk::MemoryPropertyFlagBits::eHostCoherent);
    cryptoStagingMapped = context.getDevice().mapMemory(
        cryptoStagingBuffer.getMemory(), 0,
        context.getDevice().getBufferMemoryRequirements(
            cryptoStagingBuffer.getBuffer()).size);
    positionStagingMapped = context.getDevice().mapMemory(
        positionStagingBuffer.getMemory(), 0,
        context.getDevice().getBufferMemoryRequirements(
            positionStagingBuffer.getBuffer()).size);
}

ViewportSession::~ViewportSession()
{
    context.getDevice().waitIdle();
    if (cryptoStagingMapped)
        context.getDevice().unmapMemory(cryptoStagingBuffer.getMemory());
    if (positionStagingMapped)
        context.getDevice().unmapMemory(positionStagingBuffer.getMemory());
}

// ── Layout ──────────────────────────────────────────────────────────────────

void ViewportSession::setTargetSize(const uint32_t width, const uint32_t height)
{
    if (width == 0 || height == 0)
        return;
    targetWidth = width;
    targetHeight = height;
    if (sizing != ViewportSizing::MatchTarget)
        return;

    requestedWidth = scaledExtent(static_cast<float>(width) * resolutionScale);
    requestedHeight = scaledExtent(static_cast<float>(height) * resolutionScale);
}

void ViewportSession::setResolutionScale(const float scale)
{
    resolutionScale = std::clamp(scale, 0.1f, 2.0f);
    setTargetSize(targetWidth, targetHeight);
}

void ViewportSession::setSizing(const ViewportSizing value)
{
    sizing = value;
    setTargetSize(targetWidth, targetHeight);
}

void ViewportSession::setSensorFit(const SensorFit fit)
{
    sensorFit = fit;
    applySensorFit();
    scene.setDirtyFlag(Accumulation);
    scene.setDirtyFlag(CameraState);
}

ViewportLayout ViewportSession::layout() const
{
    const auto available = glm::vec2(targetWidth, targetHeight);
    if (available.x <= 0.0f || available.y <= 0.0f || raytracer->getHeight() == 0)
        return {};

    // Under MatchTarget the render extent is the rect, so the image fills it
    // one-to-one and there is nothing to crop or letterbox. Only the fixed
    // resolution mode, where the camera dictates an extent the rect cannot
    // match, still has to fit the image into the region.
    if (sizing == ViewportSizing::MatchTarget)
        return {0.0f, 0.0f, available.x, available.y, 0.0f, 0.0f, 1.0f, 1.0f};

    const float renderAspect = static_cast<float>(raytracer->getWidth())
        / static_cast<float>(raytracer->getHeight());
    const float targetAspect = available.x / available.y;
    if (fill == ViewportFill::Cover)
    {
        ViewportLayout result{0.0f, 0.0f, available.x, available.y, 0.0f, 0.0f, 1.0f, 1.0f};
        if (targetAspect > renderAspect)
        {
            const float visible = renderAspect / targetAspect;
            result.uvMinY = (1.0f - visible) * 0.5f;
            result.uvMaxY = result.uvMinY + visible;
        }
        else if (targetAspect < renderAspect)
        {
            const float visible = targetAspect / renderAspect;
            result.uvMinX = (1.0f - visible) * 0.5f;
            result.uvMaxX = result.uvMinX + visible;
        }
        return result;
    }

    glm::vec2 size = available;
    if (targetAspect > renderAspect)
        size.x = available.y * renderAspect;
    else
        size.y = available.x / renderAspect;
    return {(available.x - size.x) * 0.5f, (available.y - size.y) * 0.5f, size.x, size.y,
        0.0f, 0.0f, 1.0f, 1.0f};
}

void ViewportSession::setFill(const ViewportFill value) { fill = value; }

uint32_t ViewportSession::renderWidth() const { return raytracer->getWidth(); }
uint32_t ViewportSession::renderHeight() const { return raytracer->getHeight(); }

void ViewportSession::applySensorFit()
{
    CameraInstance* camera = scene.getRenderCamera();
    if (!camera)
        return;
    Sensor& sensor = camera->getCamera()->getSensor();
    sensor.setResolution(raytracer->getWidth(), raytracer->getHeight());
    // Without this the film keeps the sensor's own aspect ratio while the
    // render target has another, and both the traced image and the projection
    // matrix used for gizmos come out stretched.
    sensor.setFilmFit(sensorFit, raytracer->getWidth(), raytracer->getHeight());
    // The GPU camera is only rebuilt from the sensor when CameraState is dirty.
    // Resizing alone does not set it, so without this the new film dimensions
    // would not reach the ray generation until something else moved the camera
    // - which is exactly the "aspect ratio is sometimes wrong" failure.
    scene.setDirtyFlag(CameraState);
    scene.setDirtyFlag(Accumulation);
}

void ViewportSession::ensureDisplayImage(const uint32_t width, const uint32_t height)
{
    if (displayImage.getImage() && displayImage.getWidth() == width
        && displayImage.getHeight() == height)
        return;

    // Retire rather than destroy: record() rescales the outgoing frame into the
    // new image, so there is always something to show while the first render at
    // the new size is still in flight. Destroying it here instead is what made a
    // resize flash black. resizeRenderTarget() has already waited for the device
    // to go idle, so the retired image is guaranteed to be free of GPU use.
    if (displayImage.getImage())
    {
        retiredDisplayImage = std::move(displayImage);
        carryOverPending = true;
    }
    displayImage = Image(context, width, height, displayFormat,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst
            | vk::ImageUsageFlagBits::eTransferSrc);
    displayImageCleared = false;
    ++displayImageGeneration;
}

void ViewportSession::resizeRenderTarget(const uint32_t width, const uint32_t height)
{
    if (width == 0 || height == 0
        || (width == raytracer->getWidth() && height == raytracer->getHeight()))
        return;

    context.getDevice().waitIdle();
    raytracer->waitForRender();

    raytracer->resize(width, height);
    applySensorFit();
    compositor->resize(
        raytracer->getWidth(), raytracer->getHeight(),
        raytracer->getOutputColor(), raytracer->getOutputAlbedo(),
        raytracer->getOutputNormal(), raytracer->getOutputCrypto(),
        raytracer->getOutputPosition(), raytracer->getOutputDenoised(),
        displayFormat);
    ensureDisplayImage(raytracer->getWidth(), raytracer->getHeight());

    frameIndex = 0;
    submittedSamples = 0;
    renderComplete = false;
    firstFrame = true;
    pendingReadyValue = 0;
    displayedRenderValue = 0;
}

void ViewportSession::applyPendingResize()
{
    if (sizing != ViewportSizing::MatchTarget)
    {
        // The camera drives the resolution; follow whatever it asks for.
        if (const CameraInstance* camera = scene.getRenderCamera())
        {
            const glm::uvec2 resolution =
                camera->getCamera()->getSensor().resolution();
            resizeRenderTarget(resolution.x, resolution.y);
        }
        return;
    }

    if (requestedWidth == 0 || requestedHeight == 0)
        return;
    // Never recycle a buffer that holds a finished frame nobody has seen yet.
    // Raytracer::resize() synchronizes the stream, so the render in flight does
    // complete - reallocating on top of it simply discards the result. Doing
    // that every frame of a drag is why the viewport went black: work was being
    // finished continuously and thrown away before it could ever be presented.
    // record() shows it on this frame and the resize lands on the next one.
    if (pendingReadyValue != 0)
        return;
    // The render target is otherwise the visible rect, applied the frame it changes.
    resizeRenderTarget(requestedWidth, requestedHeight);
}

// ── Frame loop ──────────────────────────────────────────────────────────────

void ViewportSession::update()
{
    // The raytracer hands a completed frame over exactly once.
    if (pendingReadyValue == 0)
    {
        if (const std::optional<InteropFrame> completed = raytracer->consumeInteropFrame())
        {
            pendingReadyValue = completed->readyValue;
            pendingRenderReady = completed->renderReadySemaphore;
            pendingBufferReleased = completed->bufferReleasedSemaphore;
        }
    }
    // Resize only once the frame just adopted is safely on its way to the
    // screen. This has to happen after the adopt above, not before it, or
    // pendingReadyValue is still zero and the guard never sees the frame it is
    // meant to protect.
    applyPendingResize();

    // An adopted frame belongs to the handoff in record(); queueing another
    // render before that is submitted would race the buffer.
    if (pendingReadyValue != 0 || raytracer->isRenderInFlight())
        return;

    if (scene.isDirty(Accumulation) || firstFrame)
    {
        frameIndex = 0;
        submittedSamples = 0;
        renderComplete = false;
    }
    if (renderComplete)
        return;

    const RenderSettings& settings = scene.getRenderSettings();
    const auto samples = static_cast<uint32_t>(std::max(1, settings.samples));
    const auto maxSamples = static_cast<uint32_t>(std::max(1, settings.maxSamples));
    raytracer->renderFrame(frameIndex, submittedSamples);
    submittedSamples = std::min(submittedSamples + samples, maxSamples);
    renderComplete = submittedSamples >= maxSamples;
    ++frameIndex;
    firstFrame = false;
}

ViewportFrameSync ViewportSession::record(const vk::CommandBuffer commandBuffer)
{
    const uint32_t selectedCryptomatteId =
        scene.getActiveCryptomatteId(gaussianSelection);
    // Refresh the overlay when the selection changes so the outline updates
    // without waiting for the next completed render. Not while a render is in
    // flight: the compositing pass reads the CUDA-shared output images, which
    // OptiX is writing through its surface objects until the handoff completes.
    const bool selectionChanged = displayedRenderValue != 0
        && selectedCryptomatteId != displayedCryptomatteId
        && !raytracer->isRenderInFlight();
    const bool hasFrame = pendingReadyValue != 0;
    if (!hasFrame && !selectionChanged && displayImageCleared)
        return {};

    if (!displayImageCleared)
    {
        displayImage.setImageLayout(commandBuffer, vk::ImageLayout::eTransferDstOptimal);
        if (carryOverPending && retiredDisplayImage.getImage())
        {
            // Rescale the previous frame into the new extent. It is the wrong
            // size for an instant, which reads as the image stretching with the
            // rect - far better than the black flash that clearing produces, and
            // the next completed render replaces it outright.
            retiredDisplayImage.setImageLayout(
                commandBuffer, vk::ImageLayout::eTransferSrcOptimal);
            vk::ImageBlit blit{};
            blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
            blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
            blit.srcOffsets[1] = vk::Offset3D{
                static_cast<int32_t>(retiredDisplayImage.getWidth()),
                static_cast<int32_t>(retiredDisplayImage.getHeight()), 1};
            blit.dstOffsets[1] = vk::Offset3D{
                static_cast<int32_t>(displayImage.getWidth()),
                static_cast<int32_t>(displayImage.getHeight()), 1};
            commandBuffer.blitImage(
                retiredDisplayImage.getImage(), vk::ImageLayout::eTransferSrcOptimal,
                displayImage.getImage(), vk::ImageLayout::eTransferDstOptimal,
                blit, vk::Filter::eLinear);
        }
        else
        {
            const vk::ClearColorValue clear(std::array{0.0f, 0.0f, 0.0f, 1.0f});
            const vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
            commandBuffer.clearColorImage(displayImage.getImage(),
                vk::ImageLayout::eTransferDstOptimal, clear, range);
        }
        carryOverPending = false;
        displayImage.setImageLayout(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal);
        displayImageCleared = true;
        if (!hasFrame && !selectionChanged)
            return {};
    }

    CameraInstance* camera = scene.getRenderCamera();
    const glm::mat4 viewProjection = camera
        ? camera->getProjectionMatrix() * camera->getViewMatrix()
        : glm::mat4(1.0f);
    const RenderSettings& settings = scene.getRenderSettings();
    if (showOverlays)
        compositor->updateBillboards(scene);
    compositor->dispatch(commandBuffer, selectedCryptomatteId, viewProjection,
        camera ? camera->getCamera()->exposure : 0.0f,
        static_cast<int>(settings.bufferVisualization),
        settings.tonemappingEnabled, showOverlays);
    displayedCryptomatteId = selectedCryptomatteId;

    Image& source = compositor->getOutputImage();
    source.setImageLayout(commandBuffer, vk::ImageLayout::eTransferSrcOptimal);
    displayImage.setImageLayout(commandBuffer, vk::ImageLayout::eTransferDstOptimal);
    vk::ImageCopy copyRegion{};
    copyRegion.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.extent = vk::Extent3D{raytracer->getWidth(), raytracer->getHeight(), 1};
    commandBuffer.copyImage(source.getImage(), vk::ImageLayout::eTransferSrcOptimal,
        displayImage.getImage(), vk::ImageLayout::eTransferDstOptimal, copyRegion);
    source.setImageLayout(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal);
    displayImage.setImageLayout(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal);

    if (!hasFrame)
        return {};  // Overlay-only refresh; there is no CUDA frame to hand back.

    displayedRenderValue = pendingReadyValue;
    const ViewportFrameSync sync{
        pendingRenderReady, pendingBufferReleased, pendingReadyValue};
    pendingReadyValue = 0;
    pendingRenderReady = vk::Semaphore{};
    pendingBufferReleased = vk::Semaphore{};
    return sync;
}

Image& ViewportSession::getDisplayImage() { return displayImage; }

uint64_t ViewportSession::displayImageRevision() const { return displayImageGeneration; }
bool ViewportSession::hasPresentedFrame() const { return displayedRenderValue != 0; }
uint32_t ViewportSession::accumulatedSamples() const { return submittedSamples; }
bool ViewportSession::isAccumulationComplete() const { return renderComplete; }

void ViewportSession::requestAccumulationReset()
{
    scene.setDirtyFlag(Accumulation);
}

// ── Picking ─────────────────────────────────────────────────────────────────

std::optional<glm::ivec2> ViewportSession::targetToPixel(
    const float targetX, const float targetY) const
{
    const ViewportLayout region = layout();
    if (region.width <= 0.0f || region.height <= 0.0f)
        return std::nullopt;

    const float localX = targetX - region.offsetX;
    const float localY = targetY - region.offsetY;
    if (localX < 0.0f || localY < 0.0f || localX >= region.width || localY >= region.height)
        return std::nullopt;

    // Map through the same UV window the host samples with, so a cropped Cover
    // layout still picks the texel that is actually under the cursor. The UVs
    // are in presentation-image space, and the render occupies its top-left
    // corner, so scaling by the allocation gives the render texel directly.
    const float u = region.uvMinX + localX / region.width * (region.uvMaxX - region.uvMinX);
    const float v = region.uvMinY + localY / region.height * (region.uvMaxY - region.uvMinY);
    const float texelX = u * static_cast<float>(raytracer->getWidth());
    const float texelY = v * static_cast<float>(raytracer->getHeight());

    // The last half pixel would map to width/height, one past the final texel.
    // AOV readbacks copy from CUDA-shared images, where an out-of-bounds region
    // reads past the shared allocation.
    return glm::ivec2(
        std::clamp(static_cast<int>(texelX), 0, static_cast<int>(raytracer->getWidth()) - 1),
        std::clamp(static_cast<int>(texelY), 0, static_cast<int>(raytracer->getHeight()) - 1));
}

bool ViewportSession::readbackAovTexel(
    Image* image, const Buffer& staging, const glm::ivec2 pixel)
{
    if (!image || !image->getImage())
        return false;
    if (pixel.x < 0 || pixel.y < 0
        || static_cast<uint32_t>(pixel.x) >= image->getWidth()
        || static_cast<uint32_t>(pixel.y) >= image->getHeight())
        return false;

    // AOV images are backed by memory shared with CUDA, and this readback both
    // reads them and transitions their layout, so an OptiX launch writing to
    // them through their surface objects has to be finished first. Overlapping
    // the two faults the device, and the resulting sticky launch failure then
    // kills every later CUDA call.
    raytracer->waitForRender();

    vk::BufferImageCopy copyRegion{};
    copyRegion.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.imageOffset = vk::Offset3D{pixel.x, pixel.y, 0};
    copyRegion.imageExtent = vk::Extent3D{1, 1, 1};
    context.oneTimeSubmit([&](const vk::CommandBuffer cmd) {
        image->setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
        cmd.copyImageToBuffer(image->getImage(), vk::ImageLayout::eTransferSrcOptimal,
            staging.getBuffer(), copyRegion);
        image->setImageLayout(cmd, vk::ImageLayout::eGeneral);
    });
    return true;
}

bool ViewportSession::pickBillboard(const glm::ivec2 pixel, ViewportPick& result) const
{
    const CameraInstance* camera = scene.getRenderCamera();
    if (!camera)
        return false;

    // Match the drawn icon so hit-testing scales with the billboard.
    constexpr float pickRadius = ViewportBillboardPixelRadius;
    const auto point = glm::vec2(pixel);
    const glm::mat4 viewProjection =
        camera->getProjectionMatrix() * camera->getViewMatrix();

    std::shared_ptr<LightInstance> closest;
    float closestDistanceSquared = pickRadius * pickRadius;
    for (const auto& object : scene.getSceneObjects())
    {
        const auto light = std::dynamic_pointer_cast<LightInstance>(object);
        if (!light)
            continue;
        const glm::vec4 clip = viewProjection
            * glm::vec4(light->getWorldTransform().getPosition(), 1.0f);
        if (clip.w <= 0.0f)
            continue;  // Behind the camera.
        const glm::vec2 ndc = glm::vec2(clip) / clip.w;
        const glm::vec2 center(
            (ndc.x * 0.5f + 0.5f) * static_cast<float>(raytracer->getWidth()),
            (1.0f - (ndc.y * 0.5f + 0.5f)) * static_cast<float>(raytracer->getHeight()));
        const glm::vec2 delta = point - center;
        if (const float distanceSquared = dot(delta, delta);
            distanceSquared <= closestDistanceSquared)
        {
            closestDistanceSquared = distanceSquared;
            closest = light;
        }
    }

    if (!closest)
        return false;
    result.hit = true;
    result.object = closest->getHandle();
    result.position = closest->getWorldTransform().getPosition();
    return true;
}

ViewportPick ViewportSession::pickObject(const float targetX, const float targetY)
{
    ViewportPick result{};
    const std::optional<glm::ivec2> pixel = targetToPixel(targetX, targetY);
    if (!pixel)
        return result;

    if (showOverlays && pickBillboard(*pixel, result))
    {
        scene.setActiveObject(result.object);
        gaussianSelection = ~0u;
        return result;
    }

    if (!readbackAovTexel(&raytracer->getOutputCrypto(), cryptoStagingBuffer, *pixel))
        return result;

    uint32_t instanceId = ~0u;
    if (cryptoStagingMapped != nullptr)
        instanceId = *static_cast<const uint32_t*>(cryptoStagingMapped);

    const auto meshInstances = scene.getMeshInstances();
    if (instanceId != ~0u && instanceId < meshInstances.size())
    {
        result.hit = true;
        result.object = meshInstances[instanceId]->getHandle();
        gaussianSelection = ~0u;
        scene.setActiveObject(result.object);
    }
    else
    {
        gaussianSelection = ~0u;
        scene.clearActiveObject();
    }
    return result;
}

std::optional<glm::vec3> ViewportSession::pickPosition(
    const float targetX, const float targetY)
{
    const std::optional<glm::ivec2> pixel = targetToPixel(targetX, targetY);
    if (!pixel)
        return std::nullopt;
    if (!readbackAovTexel(&raytracer->getOutputPosition(), positionStagingBuffer, *pixel))
        return std::nullopt;
    if (positionStagingMapped == nullptr)
        return std::nullopt;

    const auto* values = static_cast<const float*>(positionStagingMapped);
    return glm::vec3(values[0], values[1], values[2]);
}

// ── Presentation ────────────────────────────────────────────────────────────

void ViewportSession::setOverlaysEnabled(const bool enabled) { showOverlays = enabled; }
bool ViewportSession::overlaysEnabled() const { return showOverlays; }

void ViewportSession::setSelectedGaussianIndex(const uint32_t index)
{
    gaussianSelection = index;
}

uint32_t ViewportSession::selectedGaussianIndex() const { return gaussianSelection; }

Raytracer& ViewportSession::getRaytracer() { return *raytracer; }
const Raytracer& ViewportSession::getRaytracer() const { return *raytracer; }
Scene& ViewportSession::getScene() { return scene; }

}  // namespace nr
