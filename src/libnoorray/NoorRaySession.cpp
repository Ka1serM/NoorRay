#include <chrono>
#include <cstdlib>
#include <cstdio>
#include "NoorRaySession.h"

#include <stdexcept>
#include <utility>
#include <vector>

#include "Raytracing/RealtimeRaytracer.h"
#include "Logging/Log.h"
#include "Camera/CameraInstance.h"
#include "Camera/RealisticCamera.h"
#include "Scene/LightInstance.h"

#include <glm/geometric.hpp>

namespace noorray
{

namespace {
void requireRealtimeRayTracing(const noorrhi::Device& device)
{
    const auto features = device.features();
    if (!features.ray_query || !features.ray_tracing)
        throw std::runtime_error(
            "Realtime NoorRay requires VK_KHR_ray_query and VK_KHR_ray_tracing_pipeline; "
            "the selected Vulkan driver does not expose both features");
}
}

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
    requireRealtimeRayTracing(*device_);
    exportViewportMemory_ = exportColorMemory;
    viewportOutputFormat_ = noorrhi::ImageFormat::Rgba32Float;
    scene_.getRenderSettings().raytracer = RaytracerType::Realtime;
    raytracer_ = std::make_unique<RealtimeRaytracer>(*device_, width, height, exportColorMemory);
    prepareViewport();
    headless_ = true;
}

NoorRaySession::NoorRaySession(noorrhi::Device& device, const uint32_t width,
    const uint32_t height, std::function<void()> onMaterialWorkDone)
    : device_(&device)
    , scene_()
    , headless_(false)
    , materialRuntime_(onMaterialWorkDone)
{
    requireRealtimeRayTracing(device);
    scene_.getRenderSettings().raytracer = RaytracerType::Realtime;
    raytracer_ = std::make_unique<RealtimeRaytracer>(device, width, height, false,
        std::move(onMaterialWorkDone));
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
    lastRender_ = {};
    lastViewport_ = {};
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
}

void NoorRaySession::setViewportAllocationLimit(const uint32_t width, const uint32_t height)
{
    raytracer().setAllocationLimit(width, height);
}

void NoorRaySession::setViewportExternalOutput(const bool enabled, const noorrhi::ImageFormat format)
{
    if (viewportExternalOutput_ == enabled && viewportOutputFormat_ == format)
        return;
    if (device_)
        device_->synchronize();
    viewport_.reset();
    viewportExternalOutput_ = enabled;
    viewportOutputFormat_ = format;
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
    lastRender_ = device_->signal();
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
    const bool includeBillboards)
{
    ViewportPick result;
    if (!raytracer_ || x >= outputWidth() || y >= outputHeight())
        return result;

    // Billboards are drawn over the render, so where one covers the pixel it
    // is what the user clicked. The viewport stamps them into its billboard-id
    // buffer; one element read answers it.
    if (includeBillboards && viewport_) {
        if (const SceneObjectHandle object = viewport_->billboardObjectAt(x, y); object.isValid()) {
            result.hit = true;
            result.object = object;
            return result;
        }
        if (const SceneObjectHandle object = viewport_->volumeObjectAt(x, y); object.isValid()) {
            result.hit = true;
            result.object = object;
            return result;
        }
    }

    const uint32_t id = raytracer_->readCryptomatteAtOutput(x, y);
    if (id == ~0u)
        return result;
    if (const SceneObject* object = scene_.findCryptomatteObject(id)) {
        result.hit = true;
        result.object = object->getHandle();
    }
    return result;
}

std::optional<glm::vec3> NoorRaySession::pickPosition(const uint32_t x, const uint32_t y)
{
    if (!raytracer_ || x >= outputWidth() || y >= outputHeight())
        return std::nullopt;
    // Over a billboard, pivot on its object.
    if (viewport_)
        if (const auto position = viewport_->billboardPositionAt(x, y))
            return position;
    // The position AOV holds no meaningful value where the camera ray missed.
    if (raytracer_->readCryptomatteAtOutput(x, y) == ~0u)
        return std::nullopt;
    const noorrhi::float4 position = raytracer_->readPositionAtOutput(x, y);
    return glm::vec3(position.x, position.y, position.z);
}


void NoorRaySession::rebuildNativeScene()
{
    if (!raytracer_)
        throw std::runtime_error("native raytracer is not available for this session");
    rebuildNativeMaterials();
    updateNativeCamera();
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
    if (syncPending || changes != 0) {
        device_->wait(lastRender_);
        device_->wait(lastViewport_);
    }

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
    if (isDirty(TLAS) || isDirty(Meshes) || isDirty(Textures) || isDirty(Materials))
    {
        const auto dbgStarted = std::chrono::steady_clock::now(); // DBGTIME
        raytracer_->publishScene(scene_);
        NR_LOG_INFO("DBGTIME publishScene " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - dbgStarted).count() << " ms"); // DBGTIME
        for (const DirtyFlag flag : {TLAS, Meshes, Textures, Materials})
            scene_.clearDirtyFlag(flag);
        changed = true;
    }

    Scene::LightIndices changedLights;
    if (isDirty(Lights))
    {
        changedLights = scene_.takeChangedLights();
        raytracer_->updateLights(scene_, changedLights);
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
        // Scene, camera, and animated-content changes invalidate only the
        // beauty running average. ReSTIR and denoiser history remain intact.
        accumulationRestarted_ = true;
        changed = true;
    }
    appliedSceneChanges_ = scene_.getChangeState();
    if (viewport_)
        viewport_->updateOverlays(scene_);
    if (changed)
        outlineHistoryRestarted_ = true;
    return changed;
}

void NoorRaySession::updateNativeCamera()
{
    if (!raytracer_)
        return;
    raytracer_->updateCamera(scene_);
}

void NoorRaySession::restartAccumulation()
{
    accumulationRestarted_ = true;
}

bool NoorRaySession::consumeAccumulationRestart()
{
    return std::exchange(accumulationRestarted_, false);
}

bool NoorRaySession::prepareViewport()
{
    if (!raytracer_)
        return false;

    // Outside any recorded frame: this may wait for the device and replace the
    // renderer's per-frame images, and it must happen before the trace size
    // below is read.
    const bool accumulationRestarts = raytracer_->prepareFrameResources();
    if (accumulationRestarts) {
        restartAccumulation();
        outlineHistoryRestarted_ = true;
    }

    auto* realtime = dynamic_cast<RealtimeRaytracer*>(raytracer_.get());
    const ViewportInputs inputs{
        raytracer_->outputTexture(), raytracer_->albedoTexture(),
        raytracer_->normalTexture(),
        realtime ? realtime->viewportCryptomatteTexture() : raytracer_->cryptomatteTexture(),
        raytracer_->positionTexture(),
        realtime ? realtime->viewportDepthTexture() : noorrhi::TextureHandle{},
        realtime ? realtime->viewportDepthJitter() : glm::vec2{}};
    if (!viewport_)
        viewport_.emplace(*device_, raytracer_->width(), raytracer_->height(),
            raytracer_->traceWidth(), raytracer_->traceHeight(),
            raytracer_->imageWidth(), raytracer_->imageHeight(), inputs,
            viewportOutputFormat_, exportViewportMemory_, viewportExternalOutput_);
    else
        viewport_->resize(raytracer_->width(), raytracer_->height(),
            raytracer_->traceWidth(), raytracer_->traceHeight(),
            raytracer_->imageWidth(), raytracer_->imageHeight(), inputs,
            viewportOutputFormat_);
    return accumulationRestarts;
}

void NoorRaySession::renderViewport(const glm::mat4& viewProjection,
    const uint32_t selectedCryptomatteId, const bool showBillboards)
{
    if (!viewport_)
        return;
    if (auto* realtime = dynamic_cast<RealtimeRaytracer*>(raytracer_.get()))
        viewport_->setDepthJitter(realtime->viewportDepthJitter());
    if (const auto* camera = scene_.getRenderCamera())
        viewport_->setBillboardCameraPosition(camera->getPosition());
    else
        viewport_->clearBillboardCameraPosition();
    const RenderSettings& settings = scene_.getRenderSettings();
    const bool restartOutline = outlineHistoryRestarted_;
    outlineHistoryRestarted_ = false;
    viewport_->dispatch(selectedCryptomatteId, restartOutline, viewProjection, 0.0f,
        static_cast<int>(settings.bufferVisualization), settings.tonemappingEnabled,
        showBillboards, scene_.getActiveObjectHandle());
    lastViewport_ = device_->signal();
}

void NoorRaySession::renderViewport(const glm::mat4& viewProjection, const ViewportOutput& output,
    const uint32_t selectedCryptomatteId, const bool showBillboards)
{
    if (!viewport_)
        return;
    if (auto* realtime = dynamic_cast<RealtimeRaytracer*>(raytracer_.get()))
        viewport_->setDepthJitter(realtime->viewportDepthJitter());
    if (const auto* camera = scene_.getRenderCamera())
        viewport_->setBillboardCameraPosition(camera->getPosition());
    else
        viewport_->clearBillboardCameraPosition();
    const RenderSettings& settings = scene_.getRenderSettings();
    const bool restartOutline = outlineHistoryRestarted_;
    outlineHistoryRestarted_ = false;
    viewport_->dispatch(selectedCryptomatteId, restartOutline, viewProjection, 0.0f,
        static_cast<int>(settings.bufferVisualization), settings.tonemappingEnabled,
        showBillboards, scene_.getActiveObjectHandle(), output);
    lastViewport_ = device_->signal();
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
    materialRuntime_.compileAndWait(scene_);

    // A published program changes its material's hit groups and its sections'
    // opacity, which the scene upload applies along with the materials.
    raytracer_->uploadScene(scene_);
    raytracer_->waitForMaterialShaders();
    raytracer_->uploadEnvironment(scene_);
}

bool NoorRaySession::processNativeMaterials()
{
    if (!raytracer_)
        return false;
    const bool linked = raytracer_->linkCompiledMaterialShaders();
    // Adopting a linked pipeline swaps default materials for shaded ones
    // without any scene change, so the beauty average is stale from here.
    if (linked)
        restartAccumulation();
    // Published programs reach the GPU with the next pollNativeScene().
    return materialRuntime_.processPending(scene_) || linked;
}

void NoorRaySession::beginNativeMaterialImport()
{
    materialRuntime_.beginImport();
}

void NoorRaySession::endNativeMaterialImport()
{
    materialRuntime_.endImport();
}

bool NoorRaySession::nativeMaterialImportReady() const
{
    return !materialRuntime_.needsCompilation(scene_)
        && (!raytracer_ || raytracer_->materialShaderStage() == MaterialShaderStage::Idle);
}

MaterialShaderStage NoorRaySession::materialShaderStage() const
{
    return raytracer_ ? raytracer_->materialShaderStage() : MaterialShaderStage::Idle;
}

}
