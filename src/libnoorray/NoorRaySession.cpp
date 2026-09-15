#include "NoorRaySession.h"

#include <stdexcept>
#include <utility>
#include <vector>

#include "Raytracing/Raytracer.h"
#include "Logging/Log.h"
#include "Camera/CameraInstance.h"
#include "Camera/RealisticCamera.h"

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
    device_.emplace();
    exportViewportMemory_ = exportColorMemory;
    viewportOutputFormat_ = noorrhi::ImageFormat::Rgba32Float;
    raytracer_.emplace(*device_, width, height, exportColorMemory);
    prepareViewport();
    headless_ = true;
}

NoorRaySession::NoorRaySession(noorrhi::SurfaceProvider& surfaceProvider)
    : scene_()
    , headless_(false)
{
    device_.emplace(noorrhi::DeviceConfig{.surface = &surfaceProvider});
    swapchain_.emplace(device_->swapchain());
    raytracer_.emplace(*device_, surfaceProvider.width(), surfaceProvider.height());
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
    swapchain_.reset();
    scene_.releaseGpuResources();
    device_.reset();
    headless_ = true;
    renderSettingsInitialized = false;
}

noorrhi::Device& NoorRaySession::device()
{
    if (!device_)
        throw std::runtime_error("graphics device is not initialized");
    return *device_;
}

noorrhi::Swapchain& NoorRaySession::swapchain()
{
    if (!swapchain_)
        throw std::runtime_error("swapchain is not initialized");
    return *swapchain_;
}

void NoorRaySession::resize(const uint32_t width, const uint32_t height)
{
    if (!raytracer_)
        throw std::runtime_error("native raytracer is not initialized");
    raytracer_->resize(width, height);
    prepareViewport();
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

noorrhi::Frame NoorRaySession::beginFrame()
{
    return device().begin_frame(swapchain());
}

void NoorRaySession::endFrame(noorrhi::Frame&& frame)
{
    device().end_frame(std::move(frame));
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

void NoorRaySession::rebuildNativeScene()
{
    if (!raytracer_)
        throw std::runtime_error("native raytracer is not available for this session");
    raytracer_->uploadScene(scene_);
    updateNativeCamera();
    rebuildNativeMaterials();
    scene_.clearDirtyFlags();
    scene_.consumeGpuSync();
}

bool NoorRaySession::pollNativeScene()
{
    if (!raytracer_)
        return false;

    // Scene edits are intentionally cheap and coalesced. Wait once here,
    // immediately before replacing the renderer's immutable GPU snapshot.
    if (scene_.consumeGpuSync())
        raytracer_->device().synchronize();

    // Render settings are small launch data, but do not rewrite them when the
    // scene_ has not changed. This also keeps the update phase genuinely dirty-
    // driven while preserving immediate visibility after an edit.
    const RenderSettings& renderSettings = scene_.getRenderSettings();
    if (!renderSettingsInitialized || appliedRenderSettings != renderSettings)
    {
        raytracer_->applyRenderSettings(renderSettings);
        appliedRenderSettings = renderSettings;
        renderSettingsInitialized = true;
    }

    bool changed = false;
    const bool geometryDirty = scene_.isDirty(TLAS)
        || scene_.isDirty(Meshes) || scene_.isDirty(GaussianData);
    if (geometryDirty)
    {
        const bool structural = scene_.isDirty(Meshes);
        const bool updateGaussians = scene_.isDirty(GaussianData);
        if (structural || !raytracer_->updateScene(scene_, updateGaussians))
            raytracer_->uploadScene(scene_);
        scene_.clearDirtyFlag(TLAS);
        scene_.clearDirtyFlag(Meshes);
        scene_.clearDirtyFlag(GaussianData);
        changed = true;
    }

    if (scene_.isDirty(Lights))
    {
        raytracer_->updateLights(scene_);
        scene_.clearDirtyFlag(Lights);
        changed = true;
    }

    if (scene_.isDirty(EnvironmentCdf))
    {
        raytracer_->uploadEnvironment(scene_);
        scene_.clearDirtyFlag(EnvironmentCdf);
        changed = true;
    }
    if (scene_.isDirty(CameraState))
    {
        updateNativeCamera();
        scene_.clearDirtyFlag(CameraState);
        changed = true;
    }

    if (changed || scene_.isDirty(Accumulation))
    {
        scene_.clearAccumulationDirtyFlag();
        changed = true;
    }
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

    const ViewportInputs inputs{
        raytracer_->outputTexture(), raytracer_->albedoTexture(),
        raytracer_->normalTexture(), raytracer_->cryptomatteTexture(),
        raytracer_->positionTexture(), raytracer_->gaussianOverdrawPtr()};
    if (!viewport_)
        viewport_.emplace(*device_, raytracer_->width(), raytracer_->height(), inputs,
            viewportOutputFormat_, exportViewportMemory_);
    else
        viewport_->resize(raytracer_->width(), raytracer_->height(), inputs,
            viewportOutputFormat_);
    viewport_->updateBillboards(scene_);
}

void NoorRaySession::renderViewport(const glm::mat4& viewProjection,
    const uint32_t selectedCryptomatteId, const bool showBillboards)
{
    if (!viewport_)
        return;
    const RenderSettings& settings = scene_.getRenderSettings();
    viewport_->dispatch(selectedCryptomatteId, viewProjection, 0.0f,
        static_cast<int>(settings.bufferVisualization),
        settings.gaussianProxyOverdrawMax, settings.tonemappingEnabled,
        showBillboards);
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

void NoorRaySession::rebuildNativeMaterials()
{
    if (!raytracer_)
        return;

    // Compilation is asynchronous so edits never race a frame.  Publishing
    // the completed table here keeps the renderer's buffers immutable between
    // dispatches while still making scene_ imports immediately renderable.
    materialRuntime_.compileAndWait(scene_);

    raytracer_->uploadMaterials(scene_);
    raytracer_->uploadEnvironment(scene_);
}

bool NoorRaySession::processNativeMaterials()
{
    if (!raytracer_)
        return false;
    const bool wasIncomplete = materialRuntime_.needsCompilation(scene_);
    materialRuntime_.processPending(scene_);
    if (!wasIncomplete || materialRuntime_.needsCompilation(scene_))
        return false;

    raytracer_->uploadMaterials(scene_);
    return true;
}

}
