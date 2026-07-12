#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>

#include "IO/Bitmap.h"
#include "Scene/Scene.h"
#include "Vulkan/Context.h"

class Raytracer;
class GaussianTrainRenderer;

namespace noorray
{

struct GaussianTrainingConfig;
struct GaussianTrainingStep;
class GaussianTrainer;

class RenderSession
{
public:
    RenderSession(uint32_t width, uint32_t height);
    ~RenderSession();

    RenderSession(const RenderSession&) = delete;
    RenderSession& operator=(const RenderSession&) = delete;

    void loadScene(const std::string& scenePath);
    void importFile(const std::string& path);
    void readScene(const std::string& path);
    void addPerspectiveCamera(const glm::vec3& position, float focalLengthMm);
    void setCameraToWorld(const glm::mat4& cameraToWorld);
    void setCameraFocalLength(float focalLengthMm);
    void setSamples(int samples);
    void setMaxSamples(int maxSamples);
    void setMaxBounces(int maxBounces);
    void setExposure(float exposure);
    Bitmap renderBitmap(uint32_t spp);
    void renderToDevice(float* rgbaDevice, uint32_t spp);
    void setGaussianOpacity(uint32_t gaussianIndex, float opacity);

    // Differentiable Gaussian-training forward render using the same
    // Raytracer OptiX module and pipeline as the interactive renderer.
    void renderTrainForward(
        const float* positionDevice, const float* logScaleDevice,
        const float* rotationDevice, const float* opacityLogitDevice,
        const float* colorRgbDevice, uint32_t gaussianCount,
        const glm::mat4& cameraToWorld, float fovYDegrees,
        uint32_t width, uint32_t height, uint32_t samplesPerPixel, uint64_t seed,
        float* outputColorDevice);

    // Gradient pass for the preceding renderTrainForward() call -- see
    // GaussianTrainRenderer::renderBackward.
    void renderTrainBackward(
        const float* positionDevice, const float* logScaleDevice,
        const float* rotationDevice, const float* opacityLogitDevice,
        const float* colorRgbDevice, uint32_t gaussianCount,
        const glm::mat4& cameraToWorld, float fovYDegrees,
        uint32_t width, uint32_t height, uint32_t samplesPerPixel, uint64_t seed,
        const float* dLdImageDevice,
        float* dPositionDevice, float* dLogScaleDevice, float* dRotationDevice,
        float* dOpacityLogitDevice, float* dColorRgbDevice);

    // Writes the given trainable parameters to a .ply/.sog file -- see
    // GaussianTrainRenderer::exportGaussians.
    void exportGaussians(
        const std::string& path,
        const float* positionDevice, const float* logScaleDevice, const float* rotationDevice,
        const float* opacityLogitDevice, const float* colorRgbDevice, uint32_t gaussianCount);

    // Decomposes the loaded scene's single GaussianAsset (v1 scope, see
    // GaussianTrainRenderer) back into trainable leaf parameters -- lets a
    // trainer bootstrap from an existing .ply/.sog via the already-robust
    // GaussianAsset importer instead of re-parsing the file in Python.
    // Each output array must hold gaussianCount() * {3,3,4,1,3} floats.
    void getGaussianTrainableParams(
        float* position, float* logScale, float* rotation,
        float* opacityLogit, float* colorRgb) const;

    void beginGaussianTraining(const GaussianTrainingConfig& config);
    void addGaussianTrainingView(const float* targetRgb, uint32_t width, uint32_t height,
        const glm::mat4& cameraToWorld, float fovYDegrees, const std::string& name = {});
    GaussianTrainingStep trainGaussianStep();
    std::vector<GaussianTrainingStep> trainGaussians(uint32_t iterations);
    void exportTrainedGaussians(const std::string& path) const;

    uint32_t width() const;
    uint32_t height() const;
    uint32_t gaussianCount() const;

private:
    Context context;
    Scene scene;
    std::unique_ptr<Raytracer> raytracer;
    std::unique_ptr<GaussianTrainRenderer> trainRenderer;
    std::unique_ptr<GaussianTrainer> gaussianTrainer;
    uint32_t requestedWidth{};
    uint32_t requestedHeight{};
};

}
