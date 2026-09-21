#include "NoorRaySession.h"

#include <stdexcept>
#include <utility>
#include <vector>

#include "Raytracing/Raytracer.h"
#include "Logging/Log.h"
#include "Camera/CameraInstance.h"
#include "Camera/RealisticCamera.h"
#include "Scene/LightInstance.h"

#include <glm/geometric.hpp>

namespace noorray
{

NoorRaySession::NoorRaySession()
    : scene_()
{}

void NoorRaySession::initializeHeadlessRenderer(const uint32_t width,
    const uint32_t height, const bool exportColorMemory)
{
    if (width == 0 || height == 0)
        throw std::invalid_argument("headless renderer dimensions must be non-zero");
    shutdownRenderer();
    ownedDevice_.emplace();
    device_ = &*ownedDevice_;
    exportViewportMemory_ = exportColorMemory;
    viewportOutputFormat_ = noorrhi::ImageFormat::Rgba32Float;
    scene_.getRenderSettings().raytracer = RaytracerType::Realtime;
    raytracer_ = Raytracer::create(*device_, width, height, exportColorMemory);
    prepareViewport();
    headless_ = true;
}

NoorRaySession::NoorRaySession(noorrhi::Device& device, const uint32_t width,
    const uint32_t height)
    : device_(&device)
    , scene_()
    , headless_(false)
{
    scene_.getRenderSettings().raytracer = RaytracerType::Realtime;
    raytracer_ = Raytracer::create(device, width, height);
    exportViewportMemory_ = false;
    viewportOutputFormat_ = noorrhi::ImageFormat::Rgba32Float;
    prepareViewport();
}

NoorRaySession::~NoorRaySession()
{
    materialRuntime_.shutdown();
}

void NoorRaySession::shutdownRenderer()
{
    if (raytracer_)
        raytracer_->device().synchronize();
    viewport_.reset();
    raytracer_.reset();
    scene_.releaseGpuResources();
    ownedDevice_.reset();
    device_ = nullptr;
    headless_ = true;
    renderSettingsInitialized = false;
}

noorrhi::Device& NoorRaySession::device()
{
    if (!device_)
        throw std::runtime_error("graphics device is not initialized");
    return *device_;
}

Raytracer& NoorRaySession::raytracer()
{
    if (!raytracer_)
        throw std::runtime_error("native raytracer is not initialized");
    return *raytracer_;
}

Viewport& NoorRaySession::viewport()
{
    if (!viewport_)
        throw std::runtime_error("viewport is not initialized");
    return *viewport_;
}

const Viewport& NoorRaySession::viewport() const
{
    if (!viewport_)
        throw std::runtime_error("viewport is not initialized");
    return *viewport_;
}

void NoorRaySession::resizeViewport(const uint32_t width, const uint32_t height)
{
    if (!raytracer_)
        throw std::runtime_error("native raytracer is not initialized");

    if (CameraInstance* camera = scene_.getRenderCamera()) {
        // The ray target and the camera film must describe the same rectangle.
        // Keep the physical sensor width/FOV stable while fitting its film
        // height to the new render aspect; otherwise the image is stretched
        // even though the Vulkan texture itself has the requested dimensions.
        Sensor& sensor = camera->getCamera()->getSensor();
        sensor.setResolution(width, height);
        sensor.setFilmFit(SensorFit::Horizontal, width, height);
        camera->markDirty();
    }

    raytracer_->resize(width, height);
    updateNativeCamera();
    prepareViewport();
}

void NoorRaySession::reserveViewport(const uint32_t width, const uint32_t height)
{
    if (!raytracer_)
        throw std::runtime_error("native raytracer is not initialized");
    raytracer_->reserve(width, height);
    prepareViewport();
}

void NoorRaySession::resize(const uint32_t width, const uint32_t height)
{
    resizeViewport(width, height);
}

void NoorRaySession::commit()
{
    if (!raytracer_)
        throw std::runtime_error("native raytracer is not initialized");
    raytracer_->commit();
}

void NoorRaySession::render(const uint32_t frameIndex, const uint32_t sampleIndex)
{
    if (!raytracer_)
        throw std::runtime_error("native raytracer is not initialized");
    raytracer_->render(frameIndex, sampleIndex);
    if (sampleIndex == 0)
        accumulationRestarted_ = true;
}

double NoorRaySession::lastDispatchMilliseconds()
{
    if (!raytracer_)
        return 0.0;
    return raytracer_->lastDispatchMilliseconds();
}

void NoorRaySession::synchronize()
{
    if (device_)
        device_->synchronize();
}

void NoorRaySession::copyOutputTo(const noorrhi::ImageHandle target)
{
    device().copy(outputImageHandle(), target);
}

std::vector<std::byte> NoorRaySession::readColor()
{
    if (!raytracer_)
        return {};
    return raytracer_->readColor();
}

std::vector<noorrhi::float4> NoorRaySession::readBeauty()
{
    if (!raytracer_)
        return {};
    return raytracer_->readBeauty();
}

std::vector<std::uint32_t> NoorRaySession::readCryptomatte()
{
    if (!raytracer_)
        return {};
    return raytracer_->readCryptomatte();
}

std::vector<noorrhi::float4> NoorRaySession::readPosition()
{
    if (!raytracer_)
        return {};
    return raytracer_->readPosition();
}

NoorRaySession::ViewportPick NoorRaySession::pick(const uint32_t x, const uint32_t y,
    const bool includeLights)
{
    ViewportPick result;
    if (!raytracer_ || x >= outputWidth() || y >= outputHeight())
        return result;

    // Light icons are drawn over the render, so where one covers the pixel it
    // is what the user clicked. The viewport stamps them into its light-id
    // buffer; one element read answers it.
    if (includeLights && viewport_) {
        if (const SceneObjectHandle light = viewport_->lightAt(x, y); light.isValid()) {
            result.hit = true;
            result.object = light;
            return result;
        }
    }

    const uint32_t id = raytracer_->readCryptomatteAt(x, y);
    if (id == ~0u)
        return result;
    if (const SceneObject* object = scene_.findCryptomatteObject(id, result.gaussianIndex)) {
        result.hit = true;
        result.object = object->getHandle();
    }
    return result;
}

std::optional<glm::vec3> NoorRaySession::pickPosition(const uint32_t x, const uint32_t y)
{
    if (!raytracer_ || x >= outputWidth() || y >= outputHeight())
        return std::nullopt;
    // Over a light icon, pivot on the light itself.
    if (viewport_)
        if (const auto light = viewport_->lightPositionAt(x, y))
            return light;
    // The position AOV holds no meaningful value where the camera ray missed.
    if (raytracer_->readCryptomatteAt(x, y) == ~0u)
        return std::nullopt;
    const noorrhi::float4 position = raytracer_->readPositionAt(x, y);
    return glm::vec3(position.x, position.y, position.z);
}

void NoorRaySession::rebuildNativeScene()
{
    if (!raytracer_)
        throw std::runtime_error("native raytracer is not available for this session");
    raytracer_->uploadScene(scene_);
    updateNativeCamera();
    rebuildNativeMaterials();
    scene_.clearDirtyFlags();
    appliedSceneChanges_ = scene_.getChangeState();
    scene_.consumeGpuSync();
}

bool NoorRaySession::pollNativeScene()
{
    if (!raytracer_)
        return false;

    // Scene edits are intentionally cheap and coalesced. Wait once here,
    // immediately before replacing the renderer's immutable GPU snapshot.
    const uint8_t changes = scene_.changesSince(appliedSceneChanges_);
    const bool syncPending = scene_.consumeGpuSync();
    if (syncPending || changes != 0)
        raytracer_->device().synchronize();

    // Render settings are small launch data, but do not rewrite them when the
    // scene_ has not changed. This also keeps the update phase genuinely dirty-
    // driven while preserving immediate visibility after an edit.
    RenderSettings& renderSettings = scene_.getRenderSettings();
    renderSettings.raytracer = RaytracerType::Realtime;
    const bool settingsChanged = !renderSettingsInitialized || appliedRenderSettings != renderSettings;
    if (settingsChanged)
    {
        raytracer_->applyRenderSettings(renderSettings);
        appliedRenderSettings = renderSettings;
        renderSettingsInitialized = true;
    }

    const auto isDirty = [changes](DirtyFlag flag) { return (changes & flag) != 0; };
    bool changed = settingsChanged;
    const bool geometryDirty = isDirty(TLAS)
        || isDirty(Meshes) || isDirty(GaussianData) || isDirty(Textures);
    if (geometryDirty)
    {
        const bool structural = isDirty(Meshes) || isDirty(Textures);
        const bool updateGaussians = isDirty(GaussianData);
        if (structural || !raytracer_->updateScene(scene_, updateGaussians))
            raytracer_->uploadScene(scene_);
        scene_.clearDirtyFlag(TLAS);
        scene_.clearDirtyFlag(Meshes);
        scene_.clearDirtyFlag(Textures);
        scene_.clearDirtyFlag(GaussianData);
        changed = true;
    }

    if (isDirty(Lights))
    {
        raytracer_->updateLights(scene_);
        scene_.clearDirtyFlag(Lights);
        changed = true;
    }

    if (isDirty(EnvironmentCdf))
    {
        raytracer_->uploadEnvironment(scene_);
        scene_.clearDirtyFlag(EnvironmentCdf);
        changed = true;
    }
    if (isDirty(CameraState))
    {
        updateNativeCamera();
        scene_.clearDirtyFlag(CameraState);
        changed = true;
    }

    if (changed || isDirty(Accumulation))
    {
        scene_.clearAccumulationDirtyFlag();
        changed = true;
    }
    appliedSceneChanges_ = scene_.getChangeState();
    return changed;
}

void NoorRaySession::updateNativeCamera()
{
    if (!raytracer_)
        return;
    raytracer_->updateCamera(scene_);
}

void NoorRaySession::prepareViewport()
{
    if (!raytracer_)
        return;

    // Outside any recorded frame: this may wait for the device and replace the
    // renderer's per-frame images, and it must happen before the trace size
    // below is read.
    raytracer_->prepareFrameResources();

    const ViewportInputs inputs{
        raytracer_->outputTexture(), raytracer_->albedoTexture(),
        raytracer_->normalTexture(), raytracer_->cryptomatteTexture(),
        raytracer_->positionTexture(), raytracer_->gaussianOverdrawPtr()};
    if (!viewport_)
        viewport_.emplace(*device_, raytracer_->width(), raytracer_->height(),
            raytracer_->traceWidth(), raytracer_->traceHeight(),
            raytracer_->imageWidth(), raytracer_->imageHeight(), inputs,
            viewportOutputFormat_, exportViewportMemory_);
    else
        viewport_->resize(raytracer_->width(), raytracer_->height(),
            raytracer_->traceWidth(), raytracer_->traceHeight(),
            raytracer_->imageWidth(), raytracer_->imageHeight(), inputs,
            viewportOutputFormat_);
    viewport_->updateBillboards(scene_);
}

void NoorRaySession::renderViewport(const glm::mat4& viewProjection,
    const uint32_t selectedCryptomatteId, const bool showBillboards)
{
    if (!viewport_)
        return;
    const RenderSettings& settings = scene_.getRenderSettings();
    const bool restartOutline = accumulationRestarted_;
    accumulationRestarted_ = false;
    viewport_->dispatch(selectedCryptomatteId, restartOutline, viewProjection, 0.0f,
        static_cast<int>(settings.bufferVisualization),
        settings.gaussianProxyOverdrawMax, settings.tonemappingEnabled,
        showBillboards, scene_.getActiveObjectHandle());
}

void NoorRaySession::renderViewport(const uint32_t selectedCryptomatteId,
    const bool showBillboards)
{
    glm::mat4 viewProjection(1.0f);
    if (const CameraInstance* camera = scene_.getRenderCamera())
        viewProjection = camera->getProjectionMatrix() * camera->getViewMatrix();
    renderViewport(viewProjection, selectedCryptomatteId, showBillboards);
}

std::vector<noorrhi::float4> NoorRaySession::readOutput() const
{
    return viewport_ ? viewport_->readOutput() : std::vector<noorrhi::float4>{};
}

noorrhi::ImageHandle NoorRaySession::outputImageHandle() const
{
    return viewport_ ? viewport_->outputImageHandle() : noorrhi::ImageHandle{};
}

noorrhi::TextureHandle NoorRaySession::outputTexture() const
{
    return viewport_ ? viewport_->outputTexture() : noorrhi::TextureHandle{};
}

noorrhi::TextureHandle NoorRaySession::outputStorageTexture() const
{
    return viewport_ ? viewport_->outputStorageTexture() : noorrhi::TextureHandle{};
}

noorrhi::ImageFormat NoorRaySession::outputFormat() const
{
    return viewport_ ? viewport_->outputFormat() : noorrhi::ImageFormat::Auto;
}

uint32_t NoorRaySession::outputWidth() const
{
    return viewport_ ? viewport_->outputWidth() : 0u;
}

uint32_t NoorRaySession::outputHeight() const
{
    return viewport_ ? viewport_->outputHeight() : 0u;
}

uint32_t NoorRaySession::outputImageWidth() const
{
    return viewport_ ? viewport_->outputImageWidth() : 0u;
}

uint32_t NoorRaySession::outputImageHeight() const
{
    return viewport_ ? viewport_->outputImageHeight() : 0u;
}

void NoorRaySession::rebuildNativeMaterials()
{
    if (!raytracer_)
        return;

    // Compilation is asynchronous so edits never race a frame.  Publishing
    // the completed table here keeps the renderer's buffers immutable between
    // dispatches while still making scene_ imports immediately renderable.
    materialRuntime_.compileAndWait(scene_, raytracer_->needsSvmPrograms());

    raytracer_->uploadMaterials(scene_);
    raytracer_->uploadEnvironment(scene_);
}

bool NoorRaySession::processNativeMaterials()
{
    if (!raytracer_)
        return false;
    const bool wasIncomplete = materialRuntime_.needsCompilation(scene_);
    materialRuntime_.processPending(scene_, raytracer_->needsSvmPrograms());
    if (!wasIncomplete || materialRuntime_.needsCompilation(scene_))
        return false;

    raytracer_->uploadMaterials(scene_);
    return true;
}

}
