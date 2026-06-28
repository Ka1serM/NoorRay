#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "Camera/Camera.h"
#include "Kernels/Types.h"
#include "Kernels/Output.h"
#include "Scene/SceneTypes.h"

struct GpuMaterial
{
    glm::vec3 albedo;
    int albedoIndex;
    float specular;
    float metallic;
    float roughness;
    float ior;
    int specularIndex;
    int metallicIndex;
    int roughnessIndex;
    int normalIndex;
    glm::vec3 transmissionColor;
    int emissionIndex;
    float transmission;
    int transmissionIndex;
    glm::vec3 emission;
    float emissionStrength;
    int opacityIndex;
    float opacity;
    int _pad0;
    int _pad1;
};

struct GpuMesh
{
    const Vertex* vertices{};
    const uint32_t* indices{};
    const Face* faces{};
    const GpuMaterial* materials{};
    uint32_t vertexCount{};
    uint32_t triangleCount{};
    uint32_t materialCount{};
    uint32_t _pad0{};
};

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
    const GpuMesh* meshes{};
    const GpuInstance* instances{};
    const LightGpu* lights{};
    const cudaTextureObject_t* textures{};
    const RenderSettings* renderSettings{};
    const EnvironmentSettings* environment{};
    const Camera* camera{};
    uint32_t meshCount{};
    uint32_t instanceCount{};
    uint32_t lightCount{};
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
