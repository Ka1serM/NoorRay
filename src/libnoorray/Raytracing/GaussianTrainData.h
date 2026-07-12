#pragma once

#include <cstdint>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <optix.h>

enum class RenderMode : uint32_t
{
    Forward,
    Backward,
};

struct RayAdjoint
{
    glm::vec3 dOrigin{};
    glm::vec3 dDirection{};
    float dWeight{};
};

struct GaussianRayTape
{
    glm::vec3 origin{};
    glm::vec3 direction{};
    float weight{1.0f};
    uint32_t sampleKey{};
};

// Optional training payload carried by the normal wavefront launch parameters.
// The interactive path leaves this zeroed and never reads it.
struct GaussianTrainParams
{
    uint32_t enabled{};
    RenderMode mode{RenderMode::Forward};
    OptixTraversableHandle tlas{};

    // Per-Gaussian leaf parameters. Raw pointers into caller-owned (PyTorch)
    // CUDA memory -- NOT allocated/freed by this struct.
    const glm::vec3* position{};    // world-space mean
    const glm::vec3* logScale{};    // log of per-axis object-space sigma
    const glm::vec4* rotation{};    // unit quaternion (x, y, z, w)
    const float* opacityLogit{};    // opacity = sigmoid(opacityLogit)
    const glm::vec3* colorRgb{};    // linear RGB, SH degree 0 only
    uint32_t gaussianCount{};

    // Pinhole camera, decoupled from Scene/CameraInstance on purpose (see plan).
    glm::mat4 cameraToWorld{1.0f};
    float fx{1.0f}, fy{1.0f}, cx{}, cy{};

    uint32_t width{};
    uint32_t height{};
    uint32_t samplesPerPixel{1};   // M_f (forward) or M_b (backward)
    uint64_t seed{};

    // Forward output: linear RGB per pixel, no tonemap/spectral conversion.
    glm::vec3* outputColor{};

    // Backward: dL/dImage in, per-Gaussian gradients out (atomically
    // accumulated; caller zeroes these before launch).
    const glm::vec3* dLdImage{};
    glm::vec3* dPosition{};
    glm::vec3* dLogScale{};
    glm::vec4* dRotation{};
    float* dOpacityLogit{};
    glm::vec3* dColorRgb{};

    // Cutoff sigma^2 for the proxy geometry (see GaussianProxyBlas), matching
    // whatever cutoff the proxy was built with.
    float cutoffDistanceSq{9.0f};
};
