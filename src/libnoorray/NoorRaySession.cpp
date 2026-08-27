#include "NoorRaySession.h"

#include <stdexcept>
#include <chrono>
#include <thread>
#include <vector>

#include "Backend/Vulkan/Raytracer/RaytracerRenderer.h"
#include "Log.h"
#include "Rendering/Camera/CameraInstance.h"
#include "Rendering/Camera/RealisticCamera.h"

namespace noorray
{

NoorRaySession::NoorRaySession()
    : scene()
{
}

void NoorRaySession::initializeHeadlessRenderer(const uint32_t width,
    const uint32_t height, const bool exportColorMemory)
{
    if (width == 0 || height == 0)
        throw std::invalid_argument("headless renderer dimensions must be non-zero");
    if (raytracer)
        raytracer->device().synchronize();
    swapchain.reset();
    device = std::make_unique<gpu::Device>();
    raytracer = std::make_unique<VulkanRaytracer>(*device, width, height,
        false, exportColorMemory);
    headless = true;
}

NoorRaySession::NoorRaySession(gpu::SurfaceProvider& surfaceProvider)
    : scene()
    , headless(false)
{
    device = std::make_unique<gpu::Device>(gpu::DeviceConfig{.surface = &surfaceProvider});
    swapchain = std::make_unique<gpu::Swapchain>(device->swapchain());
    raytracer = std::make_unique<VulkanRaytracer>(
        *device, surfaceProvider.width(), surfaceProvider.height());
}

NoorRaySession::~NoorRaySession() = default;

void NoorRaySession::rebuildNativeScene()
{
    if (!raytracer)
        throw std::runtime_error("Native Vulkan renderer is not available for this session");
    raytracer->uploadScene(scene);
    updateNativeCamera();
    rebuildNativeMaterials();
    scene.clearDirtyFlags();
    scene.clearDirtyMeshInstanceIndices();
    scene.clearDirtyGaussianInstanceIndices();
}

bool NoorRaySession::pollNativeScene()
{
    if (!raytracer)
        return false;

    // Render settings change no GPU resource, so no dirty flag carries them.
    // Republishing the push values every poll is a host-side struct write and
    // keeps edits such as the Gaussian shading mode, proxy overdraw, and
    // transparent background observable on the very next dispatch.
    raytracer->applyRenderSettings(scene.getRenderSettings());

    bool changed = false;
    const bool geometryDirty = scene.isDirty(TLAS)
        || scene.isDirty(Meshes) || scene.isDirty(GaussianData);
    if (geometryDirty)
    {
        const bool structural = scene.isDirty(Meshes);
        const bool updateGaussians = scene.isDirty(GaussianData)
            || !scene.getDirtyGaussianInstanceIndices().empty();
        if (structural || !raytracer->updateScene(scene, updateGaussians))
            raytracer->uploadScene(scene);
        scene.clearDirtyFlag(TLAS);
        scene.clearDirtyFlag(Meshes);
        scene.clearDirtyFlag(GaussianData);
        scene.clearDirtyMeshInstanceIndices();
        scene.clearDirtyGaussianInstanceIndices();
        changed = true;
    }

    if (scene.isDirty(Lights))
    {
        raytracer->updateLights(scene);
        scene.clearDirtyFlag(Lights);
        changed = true;
    }

    if (scene.isDirty(EnvironmentCdf))
    {
        raytracer->uploadEnvironment(scene);
        scene.clearDirtyFlag(EnvironmentCdf);
        changed = true;
    }
    if (scene.isDirty(CameraState))
    {
        updateNativeCamera();
        scene.clearDirtyFlag(CameraState);
        changed = true;
    }

    if (changed || scene.isDirty(Accumulation))
    {
        scene.clearAccumulationDirtyFlag();
        changed = true;
    }
    return changed;
}

void NoorRaySession::updateNativeCamera()
{
    if (!raytracer)
        return;
    if (const CameraInstance* cameraInstance = scene.getRenderCamera())
    {
        const Camera* camera = cameraInstance->getCamera();
        VulkanCameraSnapshot snapshot{};
        for (uint32_t row = 0; row < 4; ++row)
            for (uint32_t column = 0; column < 4; ++column)
                snapshot.cameraToWorld[row * 4u + column]
                    = camera->cameraToWorld[column][row];
        snapshot.projection = static_cast<uint32_t>(
            cameraInstance->getProjectionType());
        snapshot.sensorWidthMm = camera->getSensor().filmWidth();
        snapshot.sensorHeightMm = camera->getSensor().filmHeight();
        snapshot.focalLengthMm = camera->getFocalLengthMm();
        snapshot.focusDistanceCm = camera->getFocusDistanceCm();
        snapshot.sensorOrigin = static_cast<uint32_t>(
            camera->getSensor().origin());
        snapshot.exposure = camera->exposure;
        if (const auto* realistic = camera->CastOrNullptr<RealisticCamera>())
        {
            snapshot.apertureDiameterMm = realistic->apertureDiameterMm;
            raytracer->uploadLensSnapshot(realistic->optics);
        }
        else if (const auto* thinLens = camera->CastOrNullptr<ThinLensCamera>())
            snapshot.apertureDiameterMm = thinLens->apertureDiameterMm;
        else if (const auto* fisheye = camera->CastOrNullptr<FisheyeCamera>())
            snapshot.apertureDiameterMm = fisheye->apertureDiameterMm;
        raytracer->uploadCameraSnapshot(snapshot);
    }
}

void NoorRaySession::rebuildNativeMaterials()
{
    if (!raytracer)
        return;

    // Compilation is asynchronous so edits never race a frame.  Publishing
    // the completed table here keeps the renderer's buffers immutable between
    // dispatches while still making scene imports immediately renderable.
    materialRuntime.compilePending(scene);
    for (unsigned attempt = 0; materialRuntime.hasPendingCompilations()
        && attempt != 30000; ++attempt)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        materialRuntime.compilePending(scene);
    }
    if (materialRuntime.hasPendingCompilations())
        LOG_WARN("MaterialX compilation did not finish before the native "
            "Vulkan material snapshot deadline");

    const auto& programs = materialRuntime.programs();
    // The scene's Material records point into the complete immutable program
    // table. Uploading only the first range made every hit evaluate material
    // zero, while multi-material meshes still carried valid face slots.
    raytracer->uploadMaterials(scene, programs.words(),
        programs.textureIndices());
    publishedMaterialWordCount = programs.words().size();
    raytracer->uploadEnvironment(scene);
}

bool NoorRaySession::pollNativeMaterials()
{
    if (!raytracer)
        return false;
    materialRuntime.compilePending(scene);
    if (materialRuntime.hasPendingCompilations()
        || materialRuntime.needsCompilation(scene))
        return false;

    const auto& programs = materialRuntime.programs();
    if (programs.words().size() == publishedMaterialWordCount)
        return false;
    raytracer->uploadMaterials(scene, programs.words(),
        programs.textureIndices());
    publishedMaterialWordCount = programs.words().size();
    return true;
}

}
