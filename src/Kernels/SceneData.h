#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "Camera/Camera.h"
#include "Kernels/Samplers.h"
#include "Kernels/Types.h"
#include "Kernels/Output.h"
#include "Mesh/MeshAsset.h"
#include "Scene/Scene.h"
#include "Light/PointLight.h"
#include "Light/SpotLight.h"
#include "Light/RectLight.h"

struct GpuInstance
{
    float objectToWorld[12]{};
    float worldToObject[12]{};
    float normalToWorld[9]{};
    uint32_t meshIndex{};
    uint32_t objectIndex{};
};

struct GpuSceneData
{
    const MeshAsset* meshes{};
    const GpuInstance* instances{};
    const PointLight* pointLights{};
    const SpotLight* spotLights{};
    const RectLight* rectLights{};
    const DirectionalLight* directionalLights{};
    const cudaTextureObject_t* textures{};
    cudaTextureObject_t environmentCdf{};
    const RenderSettings* renderSettings{};
    const EnvironmentSettings* environment{};
    const Camera* camera{};
    R2Sampler sampler{};
    uint32_t meshCount{};
    uint32_t instanceCount{};
    uint32_t pointLightCount{};
    uint32_t spotLightCount{};
    uint32_t rectLightCount{};
    uint32_t directionalLightCount{};
    uint32_t textureCount{};
};

struct GpuFrameSettings
{
    int frame{};
    int isMoving{};
    int pixelSizePercent{100};
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
