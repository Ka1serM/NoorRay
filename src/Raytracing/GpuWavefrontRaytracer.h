#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.hpp>

#include "GPU/ImageInterop.h"
#include "Kernels/SceneData.h"
#include "Kernels/OptixLaunchParams.h"
#include "Kernels/cuda/AccelBuilder.h"
#include "Kernels/cuda/OptiXSetup.h"

class CameraInstance;
class Context;
class Image;
class Scene;

struct FrameInfo
{
    uint32_t bufferIndex{};
    uint64_t readyValue{};
    vk::Semaphore renderReadySemaphore{};
    vk::Semaphore bufferReleasedSemaphore{};
};

class GpuWavefrontRaytracer
{
public:
    GpuWavefrontRaytracer(Context& context, Scene& scene, uint32_t width, uint32_t height);
    ~GpuWavefrontRaytracer();

    GpuWavefrontRaytracer(const GpuWavefrontRaytracer&) = delete;
    GpuWavefrontRaytracer& operator=(const GpuWavefrontRaytracer&) = delete;

    void setup(uint32_t width, uint32_t height);
    void resize(uint32_t width, uint32_t height);
    void render(const PushData& pushData);
    void updateMeshes();
    void updateTextures();
    void updateTLAS();

    FrameInfo getFrameInfo() const;
    uint32_t getWidth() const { return width; }
    uint32_t getHeight() const { return height; }
    void debugSave(const std::string& path) const;
    Image& getOutputColor();
    Image& getOutputAlbedo();
    Image& getOutputNormal();
    Image& getOutputCrypto();
    Image& getOutputPosition();
    Image& getOutputMaterial();
    Image& getOutputImage(uint32_t bufferIndex, Aov aov);

private:
    static constexpr uint32_t MaxBounces = 66;

    struct DeviceMeshAllocation
    {
        Vertex* vertices{};
        uint32_t* indices{};
        Face* faces{};
        GpuMaterial* materials{};
    };

    Context& context;
    Scene& scene;
    uint32_t width{};
    uint32_t height{};
    cudaStream_t stream{};
    std::unique_ptr<ImageInterop> interop;
    OptiXSetup optix;
    AccelBuilder accel;
    WavefrontQueues queues{};
    GpuMesh* deviceMeshes{};
    GpuInstance* deviceInstances{};
    cudaTextureObject_t* deviceTextureObjects{};
    glm::vec4* accumulation{};
    glm::vec4* adaptiveState{};
    std::vector<DeviceMeshAllocation> meshAllocations;
    std::vector<cudaArray_t> textureArrays;
    std::vector<cudaTextureObject_t> textureObjects;
    std::array<OptixLaunchParams*, MaxBounces> extendParams{};
    std::array<OptixLaunchParams*, MaxBounces> connectParams{};
    GpuSceneData gpuScene{};
    uint32_t nextBuffer{};
    uint32_t lastLaunched{};
    uint64_t lastReadyValue{};
    uint64_t submittedFrame{};
    std::array<uint64_t, 2> lastUseValue{};

    void allocateQueues();
    void freeQueues() noexcept;
    void freeSceneData() noexcept;
    std::vector<AccelInstanceInput> buildInstanceInputs(std::vector<GpuInstance>* gpuInstances = nullptr) const;
    void uploadLaunchParams();
};
