#include "RenderSession.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <glm/gtc/quaternion.hpp>

#include "Camera/Camera.h"
#include "Camera/CameraInstance.h"
#include "Camera/PerspectiveCamera.h"
#include "CUDA/rstd/Allocator.h"
#include "Log.h"
#include "Raytracing/Raytracer.h"
#include "Mesh/GaussianAsset.h"
#include "Scene/GaussianInstance.h"
#include "Scene/SceneImporter.h"
#include "Scene/SceneReader.h"
#include "Training/GaussianTrainRenderer.h"
#include "Training/GaussianTrainer.h"

namespace noorray
{

RenderSession::RenderSession(const uint32_t width, const uint32_t height)
    : context(1, 1, true)
    , scene(context)
    , requestedWidth(width)
    , requestedHeight(height)
{
}

RenderSession::~RenderSession() = default;

void RenderSession::loadScene(const std::string& scenePath)
{
    gaussianTrainer.reset();
    trainRenderer.reset();
    raytracer.reset();
    if (SceneImporter::IsSceneFile(scenePath))
    {
        SceneReader::Read(scene, scenePath);
    }
    else
    {
        SceneImporter::ImportFile(scene, scenePath);

        if (!scene.getActiveCamera())
        {
            nr::rstd::allocator<PerspectiveCamera> allocator;
            PerspectiveCamera* camera = allocator.allocate(1);
            allocator.construct(camera);
            Camera cameraHandle(camera);
            cameraHandle.setFocalLength(50.0f);
            cameraHandle.getSensor().setResolution(requestedWidth, requestedHeight);
            scene.add(std::make_unique<CameraInstance>(
                scene, "Camera", Transform{glm::vec3(0.0f, 2.0f, 5.0f)}, cameraHandle));
        }
    }

    LOG_INFO("Scene loaded: " << scenePath << " (" << scene.getGaussianCount()
             << " gaussians, " << scene.getMeshAssets().size() << " meshes)");
    raytracer = std::make_unique<Raytracer>(context, scene);
}

void RenderSession::importFile(const std::string& path)
{
    gaussianTrainer.reset(); trainRenderer.reset(); raytracer.reset();
    SceneImporter::ImportFile(scene, path);
}

void RenderSession::readScene(const std::string& path)
{
    gaussianTrainer.reset(); trainRenderer.reset(); raytracer.reset();
    SceneReader::Read(scene, path);
    raytracer = std::make_unique<Raytracer>(context, scene);
}

void RenderSession::addPerspectiveCamera(const glm::vec3& position, const float focalLengthMm)
{
    nr::rstd::allocator<PerspectiveCamera> allocator;
    PerspectiveCamera* camera = allocator.allocate(1);
    allocator.construct(camera);
    Camera handle(camera);
    handle.setFocalLength(focalLengthMm);
    handle.getSensor().setResolution(requestedWidth, requestedHeight);
    scene.add(std::make_unique<CameraInstance>(scene, "Camera", Transform{position}, handle));
    raytracer = std::make_unique<Raytracer>(context, scene);
}

void RenderSession::setCameraToWorld(const glm::mat4& cameraToWorld)
{
    if (!scene.getActiveCamera()) throw std::runtime_error("No active camera");
    scene.getActiveCamera()->getCamera()->setCameraToWorld(cameraToWorld);
    scene.setDirtyFlag(CameraState);
    scene.setDirtyFlag(Accumulation);
}

void RenderSession::setCameraFocalLength(const float focalLengthMm)
{
    if (!scene.getActiveCamera()) throw std::runtime_error("No active camera");
    scene.getActiveCamera()->getCamera()->setFocalLength(focalLengthMm);
    scene.setDirtyFlag(CameraState);
    scene.setDirtyFlag(Accumulation);
}

void RenderSession::setSamples(const int samples) { scene.getRenderSettings().samples = std::max(1, samples); }
void RenderSession::setMaxSamples(const int samples) { scene.getRenderSettings().maxSamples = std::max(1, samples); }
void RenderSession::setMaxBounces(const int bounces) { scene.getRenderSettings().maxBounces = std::max(1, bounces); }
void RenderSession::setExposure(const float exposure) { scene.getRenderSettings().exposure = exposure; }

Bitmap RenderSession::renderBitmap(const uint32_t spp)
{
    if (!raytracer)
        throw std::runtime_error("No scene loaded");
    raytracer->setAovEnabled(false);
    return raytracer->renderOffline(spp);
}

void RenderSession::renderToDevice(float* rgbaDevice, const uint32_t spp)
{
    if (!raytracer)
        throw std::runtime_error("No scene loaded");
    raytracer->setAovEnabled(false);
    raytracer->renderOfflineToDevice(rgbaDevice, spp);
}

void RenderSession::setGaussianOpacity(uint32_t gaussianIndex, const float opacity)
{
    for (GaussianAsset& asset : scene.getGaussianAssets())
    {
        if (gaussianIndex < asset.getGaussianCount())
        {
            asset.getGaussians()[gaussianIndex].opacity = opacity;
            if (raytracer)
                raytracer->updateMeshes();
            return;
        }
        gaussianIndex -= asset.getGaussianCount();
    }
    throw std::out_of_range("Gaussian index is out of range");
}

uint32_t RenderSession::width() const
{
    return raytracer ? raytracer->getWidth() : requestedWidth;
}

uint32_t RenderSession::height() const
{
    return raytracer ? raytracer->getHeight() : requestedHeight;
}

uint32_t RenderSession::gaussianCount() const
{
    return scene.getGaussianCount();
}

void RenderSession::renderTrainBackward(
    const float* positionDevice, const float* logScaleDevice,
    const float* rotationDevice, const float* opacityLogitDevice,
    const float* colorRgbDevice, const uint32_t gaussianCount,
    const glm::mat4& cameraToWorld, const float fovYDegrees,
    const uint32_t width, const uint32_t height, const uint32_t samplesPerPixel, const uint64_t seed,
    const float* dLdImageDevice,
    float* dPositionDevice, float* dLogScaleDevice, float* dRotationDevice,
    float* dOpacityLogitDevice, float* dColorRgbDevice)
{
    if (!trainRenderer)
        throw std::runtime_error("renderTrainBackward called before any renderTrainForward call");
    if (auto* cameraInstance = scene.getActiveCamera())
    {
        Camera& camera = *cameraInstance->getCamera();
        camera.setCameraToWorld(cameraToWorld);
        if (fovYDegrees > 0.0f)
            camera.setFocalLength(camera.focalLengthForFov(fovYDegrees));
    }
    const float fy = 0.5f * static_cast<float>(height)
        / std::tan(glm::radians(fovYDegrees) * 0.5f);
    const GaussianTrainingCamera camera{cameraToWorld, fy, fy,
        0.5f * static_cast<float>(width), 0.5f * static_cast<float>(height)};
    trainRenderer->renderBackward(
        positionDevice, logScaleDevice, rotationDevice, opacityLogitDevice, colorRgbDevice,
        gaussianCount, camera, width, height, samplesPerPixel, seed,
        dLdImageDevice, dPositionDevice, dLogScaleDevice, dRotationDevice, dOpacityLogitDevice, dColorRgbDevice);
}

void RenderSession::exportGaussians(
    const std::string& path,
    const float* positionDevice, const float* logScaleDevice, const float* rotationDevice,
    const float* opacityLogitDevice, const float* colorRgbDevice, const uint32_t gaussianCount)
{
    if (!raytracer)
        throw std::runtime_error("No scene loaded");
    if (!trainRenderer)
        trainRenderer = std::make_unique<GaussianTrainRenderer>(*raytracer, scene);
    trainRenderer->exportGaussians(
        path, positionDevice, logScaleDevice, rotationDevice, opacityLogitDevice, colorRgbDevice, gaussianCount);
}

void RenderSession::beginGaussianTraining(const GaussianTrainingConfig& config)
{
    if (!raytracer) throw std::runtime_error("No scene loaded");
    if (!trainRenderer) trainRenderer = std::make_unique<GaussianTrainRenderer>(*raytracer, scene);
    const uint32_t count = gaussianCount();
    std::vector<float> position(count * 3), logScale(count * 3), rotation(count * 4);
    std::vector<float> opacity(count), color(count * 3);
    getGaussianTrainableParams(position.data(), logScale.data(), rotation.data(), opacity.data(), color.data());
    gaussianTrainer = std::make_unique<GaussianTrainer>(*trainRenderer, count,
        position.data(), logScale.data(), rotation.data(), opacity.data(), color.data(), config);
}

void RenderSession::addGaussianTrainingView(const float* targetRgb, const uint32_t width,
    const uint32_t height, const glm::mat4& cameraToWorld, const float fovYDegrees, const std::string& name)
{
    if (!gaussianTrainer) throw std::runtime_error("beginGaussianTraining must be called first");
    const float fy = 0.5f * static_cast<float>(height)
        / std::tan(glm::radians(fovYDegrees) * 0.5f);
    gaussianTrainer->addView(targetRgb, width, height,
        GaussianTrainingCamera{cameraToWorld, fy, fy,
            0.5f * static_cast<float>(width), 0.5f * static_cast<float>(height)}, name);
}

GaussianTrainingStep RenderSession::trainGaussianStep()
{
    if (!gaussianTrainer) throw std::runtime_error("beginGaussianTraining must be called first");
    return gaussianTrainer->trainStep();
}

std::vector<GaussianTrainingStep> RenderSession::trainGaussians(const uint32_t iterations)
{
    if (!gaussianTrainer) throw std::runtime_error("beginGaussianTraining must be called first");
    return gaussianTrainer->train(iterations);
}

void RenderSession::exportTrainedGaussians(const std::string& path) const
{
    if (!gaussianTrainer) throw std::runtime_error("beginGaussianTraining must be called first");
    gaussianTrainer->exportGaussians(path);
}

void RenderSession::getGaussianTrainableParams(
    float* position, float* logScale, float* rotation, float* opacityLogit, float* colorRgb) const
{
    const auto instances = scene.getGaussianInstances();
    if (instances.size() != 1 || scene.getGaussianAssets().size() != 1)
    {
        throw std::runtime_error(
            "getGaussianTrainableParams: scene must have exactly one Gaussian instance/asset");
    }
    const GaussianAsset& asset = scene.getGaussianAsset(instances.front()->getGaussianAssetIndex());
    const auto& gaussians = asset.getGaussians();
    for (size_t i = 0; i < gaussians.size(); ++i)
    {
        const Gaussian& g = gaussians[i];
        const glm::vec3 col0 = g.transform[0];
        const glm::vec3 col1 = g.transform[1];
        const glm::vec3 col2 = g.transform[2];
        const glm::vec3 pos  = g.transform[3];
        const float sx = glm::length(col0);
        const float sy = glm::length(col1);
        const float sz = glm::length(col2);
        const glm::mat3 R(
            sx > 0.0f ? col0 / sx : col0,
            sy > 0.0f ? col1 / sy : col1,
            sz > 0.0f ? col2 / sz : col2);
        const glm::quat q = glm::normalize(glm::quat_cast(R));

        position[i * 3 + 0] = pos.x;
        position[i * 3 + 1] = pos.y;
        position[i * 3 + 2] = pos.z;
        logScale[i * 3 + 0] = std::log(std::max(sx, 1.0e-8f));
        logScale[i * 3 + 1] = std::log(std::max(sy, 1.0e-8f));
        logScale[i * 3 + 2] = std::log(std::max(sz, 1.0e-8f));
        rotation[i * 4 + 0] = q.x;
        rotation[i * 4 + 1] = q.y;
        rotation[i * 4 + 2] = q.z;
        rotation[i * 4 + 3] = q.w;
        const float op = std::clamp(g.opacity, 1.0e-6f, 1.0f - 1.0e-6f);
        opacityLogit[i] = std::log(op / (1.0f - op));
        constexpr float C0 = 0.28209479177387814f;
        const glm::vec3 rgb = g.shCoeffCount == 0
            ? glm::vec3(0.5f)
            : glm::max(glm::vec3(0.5f) + C0 * g.shCoeffs[0], glm::vec3(0.0f));
        colorRgb[i * 3 + 0] = rgb.x;
        colorRgb[i * 3 + 1] = rgb.y;
        colorRgb[i * 3 + 2] = rgb.z;
    }
}

void RenderSession::renderTrainForward(
    const float* positionDevice, const float* logScaleDevice,
    const float* rotationDevice, const float* opacityLogitDevice,
    const float* colorRgbDevice, const uint32_t gaussianCount,
    const glm::mat4& cameraToWorld, const float fovYDegrees,
    const uint32_t width, const uint32_t height, const uint32_t samplesPerPixel, const uint64_t seed,
    float* outputColorDevice)
{
    if (!raytracer)
        throw std::runtime_error("No scene loaded");
    if (!trainRenderer)
        trainRenderer = std::make_unique<GaussianTrainRenderer>(*raytracer, scene);
    const float fy = 0.5f * static_cast<float>(height)
        / std::tan(glm::radians(fovYDegrees) * 0.5f);
    const GaussianTrainingCamera camera{cameraToWorld, fy, fy,
        0.5f * static_cast<float>(width), 0.5f * static_cast<float>(height)};
    trainRenderer->renderForward(
        positionDevice, logScaleDevice, rotationDevice, opacityLogitDevice, colorRgbDevice,
        gaussianCount, camera, width, height, samplesPerPixel, seed,
        outputColorDevice);
}

}
