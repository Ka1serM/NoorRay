#pragma once

#include <cstdint>
#include <string>

#include <cuda_runtime_api.h>
#include <glm/mat4x4.hpp>

class Raytracer;
class Scene;
namespace noorray { struct GaussianTrainingCamera; }

// Differentiable Gaussian-splat adapter over Raytracer's existing OptiX
// module, pipeline, TLAS, and stream.
//
// v1 scope: exactly one GaussianInstance/GaussianAsset in the scene (see
// bakeTransforms()). Multi-instance Gaussian scenes are not yet supported.
class GaussianTrainRenderer
{
public:
    GaussianTrainRenderer(Raytracer& raytracer, Scene& scene);
    ~GaussianTrainRenderer();

    GaussianTrainRenderer(const GaussianTrainRenderer&) = delete;
    GaussianTrainRenderer& operator=(const GaussianTrainRenderer&) = delete;

    // Renders one image from an arbitrary camera pose against the trainable
    // Gaussian parameters. All *Device pointers are raw CUDA device
    // pointers (e.g. from a PyTorch tensor's .data_ptr()), not owned here.
    // cameraToWorld is a 4x4 column-major matrix. outputColorDevice must
    // point at width*height*3 floats.
    void renderForward(
        const float* positionDevice, const float* logScaleDevice,
        const float* rotationDevice, const float* opacityLogitDevice,
        const float* colorRgbDevice, uint32_t gaussianCount,
        const noorray::GaussianTrainingCamera& camera,
        uint32_t width, uint32_t height, uint32_t samplesPerPixel, uint64_t seed,
        float* outputColorDevice);

    // Gradient pass (paper Algorithm 2). Must be called with the exact same
    // Gaussian parameters/camera/width/height/seed as the preceding
    // renderForward() call (the TLAS built by that call is reused; this
    // does NOT re-bake transforms or rebuild the TLAS). dLdImageDevice is
    // (height*width*3) floats. The five d* output buffers are zeroed here
    // and must each point at gaussianCount * {3,3,4,1,3} floats.
    void renderBackward(
        const float* positionDevice, const float* logScaleDevice,
        const float* rotationDevice, const float* opacityLogitDevice,
        const float* colorRgbDevice, uint32_t gaussianCount,
        const noorray::GaussianTrainingCamera& camera,
        uint32_t width, uint32_t height, uint32_t samplesPerPixel, uint64_t seed,
        const float* dLdImageDevice,
        float* dPositionDevice, float* dLogScaleDevice, float* dRotationDevice,
        float* dOpacityLogitDevice, float* dColorRgbDevice);

    // Writes the current trainable parameters to a .ply or .sog file (via
    // GaussForge's writers), converting back from NoorRay's OpenGL world
    // space to the raw-3DGS (Y-down, Z-forward) convention GaussianAsset
    // import assumes for files with no explicit coordinate metadata.
    void exportGaussians(
        const std::string& path,
        const float* positionDevice, const float* logScaleDevice, const float* rotationDevice,
        const float* opacityLogitDevice, const float* colorRgbDevice, uint32_t gaussianCount);

    void syncScene(
        const float* positionDevice, const float* logScaleDevice, const float* rotationDevice,
        const float* opacityLogitDevice, const float* colorRgbDevice, uint32_t gaussianCount);

private:
    // Writes the render-ready instance transform (R*S+pos) for every
    // Gaussian into the scene's single GaussianAsset and refits the TLAS via
    // Raytracer::updateTLAS() (existing, unmodified).
    void bakeTransformsAndUpdateTlas(
        const float* positionDevice, const float* logScaleDevice,
        const float* rotationDevice, uint32_t gaussianCount);

    Raytracer& raytracer;
    Scene& scene;
    cudaStream_t stream{};

};
