#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "Camera/Camera.h"
#include "CUDA/Texture.h"
#include "Raytracing/Output.h"
#include "Raytracing/Types.h"
#include "Scene/GpuInstance.h"
#include "Scene/RenderSettings.h"
#include "Mesh/MeshAsset.h"
#include "Scene/Environment.h"
#include "Light/DirectionalLight.h"
#include "Light/PointLight.h"
#include "Light/SpotLight.h"
#include "Light/RectLight.h"

struct GpuSceneData
{
    const MeshAsset* meshes{};
    const GpuInstance* instances{};
    const PointLight* pointLights{};
    const SpotLight* spotLights{};
    const RectLight* rectLights{};
    const DirectionalLight* directionalLights{};
    const CudaTexture* textures{};
    TlasHandle tlasHandle{};
    RenderSettings renderSettings{};
    const Environment* environment{};
    const Camera* camera{};
    // Jakob & Hanika sRGB→spectrum table (64^3, 9 MB on device).
    const float* spectrumTableScale{};   // 64 floats (z-nodes, non-uniform)
    const float* spectrumTableCoeffs{};  // float[3][64][64][64][3]
    const float* d65{};                  // CIE D65, 300--830 nm at 5 nm
    // CIE 1931 2-degree CMFs (471 floats each, device pointers).
    const float* cieX{};
    const float* cieY{};
    const float* cieZ{};
    uint32_t meshCount{};
    uint32_t instanceCount{};
    uint32_t pointLightCount{};
    uint32_t spotLightCount{};
    uint32_t rectLightCount{};
    uint32_t directionalLightCount{};
    uint32_t textureCount{};
    uint32_t _pad{};
};

struct GpuFrameSettings
{
    int frame{};
    int isMoving{};
    uint32_t width{};
    uint32_t height{};
    uint32_t sampleIndex{};       // which sample within this frame (RNG seed diversity)
    uint32_t totalAccumulated{};  // total samples accumulated so far (blend weight)
};

struct KernelParams
{
    GpuSceneData scene;
    WavefrontQueues queues;
    OutputSurfaces output;
    GpuFrameSettings frame;
    glm::vec4* accumulation{};
    glm::vec4* adaptiveState{};
    uint32_t depth{};
};
