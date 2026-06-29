#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <optix.h>
#include <vulkan/vulkan.hpp>

#include "GPU/ImageInterop.h"
#include "GPU/rstd/Vector.h"
#include "Kernels/SceneData.h"
#include "Kernels/cuda/TriangleBlas.h"
#include "Kernels/cuda/TlasBuilder.h"

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

struct PushData
{
    int frame{};
    int isMoving{};
    int pixelSizePercent{};
};

class Raytracer
{
public:
    Raytracer(Context& context, Scene& scene);
    ~Raytracer();

    Raytracer(const Raytracer&) = delete;
    Raytracer& operator=(const Raytracer&) = delete;

    void setup(uint32_t width, uint32_t height);
    void resize(uint32_t width, uint32_t height);
    void render(const PushData& pushData);
    void updateMeshes();
    void updateTextures();
    void updateEnvironmentCdf();
    void updateLights();
    void updateTLAS();

    FrameInfo getFrameInfo() const;
    float getGpuTimeMs() const { return m_gpuTimeMs; }
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

    Context& context;
    Scene& scene;
    uint32_t width{};
    uint32_t height{};
    cudaStream_t stream{};
    std::unique_ptr<ImageInterop> interop;
    TriangleBlas triangleBlas;
    TlasBuilder tlas;
    WavefrontQueues queues{};
    nr::rstd::vector<GpuInstance> gpuInstances;
    glm::vec4* accumulation{};
    glm::vec4* adaptiveState{};
    nr::rstd::vector<cudaArray_t> textureArrays;
    nr::rstd::vector<cudaTextureObject_t> textureObjects;
    cudaArray_t environmentCdfArray{};
    cudaTextureObject_t environmentCdfTexture{};
    GpuSceneData gpuScene{};
    uint32_t nextBuffer{};
    uint32_t lastLaunched{};
    uint64_t lastReadyValue{};
    uint64_t submittedFrame{};
    std::array<uint64_t, 2> lastUseValue{};
    cudaEvent_t m_startEvent{};
    cudaEvent_t m_stopEvent{};
    float m_gpuTimeMs = 0.0f;
    bool m_eventsRecorded = false;

    // OptiX state
    OptixDeviceContext optixCtx{};
    OptixModule optixModule{};
    OptixProgramGroup optixExtendGroup{};
    OptixProgramGroup optixConnectGroup{};
    OptixProgramGroup optixTriangleGroup{};
    OptixProgramGroup optixMissGroup{};
    OptixPipeline optixPipeline{};
    CUdeviceptr optixExtendRecord{};
    CUdeviceptr optixConnectRecord{};
    CUdeviceptr optixHitgroupRecord{};
    CUdeviceptr optixMissRecord{};
    OptixShaderBindingTable optixExtendSbt{};
    OptixShaderBindingTable optixConnectSbt{};

    void allocateQueues();
    void freeQueues() noexcept;
    void freeSceneData() noexcept;
    std::vector<AccelInstanceInput> buildInstanceInputs(
        const std::vector<OptixTraversableHandle>& meshHandles,
        nr::rstd::vector<GpuInstance>* instances = nullptr) const;
    void launchGenerate(const KernelParams& params, cudaStream_t stream) const;
    void launchFinalize(const KernelParams& params, cudaStream_t stream) const;
    void launchShade(const KernelParams& params, cudaStream_t stream) const;
    void launchExtend(const WavefrontQueues& queues, uint32_t depth, uint32_t capacity, cudaStream_t stream) const;
    void launchConnect(const WavefrontQueues& queues, uint32_t depth, uint32_t capacity, cudaStream_t stream) const;
};
