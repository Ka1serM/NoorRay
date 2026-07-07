#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <glm/vec4.hpp>
#include <optix.h>
#include <vulkan/vulkan.hpp>

#include "CUDA/KernelStats.h"
#include "CUDA/Texture.h"
#include "CUDA/Tlas.h"
#include "CUDA/SharedImage.h"
#include "IO/Bitmap.h"
#include "Raytracing/SceneData.h"

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
};

class Raytracer
{
public:
    Raytracer(Context& context, Scene& scene);
    ~Raytracer();

    Raytracer(const Raytracer&) = delete;
    Raytracer& operator=(const Raytracer&) = delete;

    void resize(uint32_t width, uint32_t height);
    void render(const PushData& pushData);
    Bitmap renderOffline(uint32_t sampleCount);

    void setAovEnabled(bool enabled) { aovEnabled = enabled; }
    bool getAovEnabled() const { return aovEnabled; }
    void setStatsEnabled(bool enabled) { kernelStats.setEnabled(enabled); }
    void harvestKernelStats() { kernelStats.harvestFrame(); }
    void printKernelStats() const { kernelStats.printReport(); }
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
    Image& getOutputColor() { return color[lastLaunched].getImage(); }
    Image& getOutputAlbedo() { return albedo[lastLaunched].getImage(); }
    Image& getOutputNormal() { return normal[lastLaunched].getImage(); }
    Image& getOutputCrypto() { return cryptomatte[lastLaunched].getImage(); }
    Image& getOutputPosition() { return position[lastLaunched].getImage(); }

    Image& getOutputColor(uint32_t bufferIndex) { return color[bufferIndex].getImage(); }
    Image& getOutputAlbedo(uint32_t bufferIndex) { return albedo[bufferIndex].getImage(); }
    Image& getOutputNormal(uint32_t bufferIndex) { return normal[bufferIndex].getImage(); }
    Image& getOutputCrypto(uint32_t bufferIndex) { return cryptomatte[bufferIndex].getImage(); }
    Image& getOutputPosition(uint32_t bufferIndex) { return position[bufferIndex].getImage(); }

private:
    static constexpr uint32_t MaxBounces = 66;

    Context& context;
    Scene& scene;
    bool aovEnabled{true};
    uint32_t width{};
    uint32_t height{};
    cudaStream_t stream{};
    SharedImage color[2];
    SharedImage albedo[2];
    SharedImage normal[2];
    SharedImage cryptomatte[2];
    SharedImage position[2];
    vk::UniqueSemaphore renderReady;
    vk::UniqueSemaphore bufferReleased;
    cudaExternalSemaphore_t cudaRenderReady{};
    cudaExternalSemaphore_t cudaBufferReleased{};
    Tlas tlas;
    WavefrontQueues queues{};
    glm::vec4* accumulation{};
    float* spectrumTableScaleDevice{};
    float* spectrumTableCoeffsDevice{};
    float* d65Device{};
    float* cieXDevice{};
    float* cieYDevice{};
    float* cieZDevice{};
    nr::openpbr::EnergyLutStorage openPbrLutStorage{};
    GpuSceneData gpuScene{};
    uint32_t nextBuffer{};
    uint32_t lastLaunched{};
    uint32_t aovStaleBuffers{2};
    uint64_t lastReadyValue{};
    uint64_t submittedFrame{};
    std::array<uint64_t, 2> lastUseValue{};
    cudaEvent_t m_startEvent{};
    cudaEvent_t m_stopEvent{};
    float m_gpuTimeMs = 0.0f;
    bool m_eventsRecorded = false;
    KernelStats kernelStats;
    CUdeviceptr optixLaunchParamsDevice{};

    // OptiX state
    OptixDeviceContext optixCtx{};
    OptixModule optixModule{};
    OptixProgramGroup optixExtendGroup{};
    OptixProgramGroup optixConnectGroup{};
    OptixProgramGroup optixTriangleGroup{};
    OptixProgramGroup optixGaussianHitGroup{};
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
    void launchGenerate(const KernelParams& params, cudaStream_t stream) const;
    void launchFinalize(const KernelParams& params, cudaStream_t stream) const;
    void launchShade(const KernelParams& params, uint32_t launchCount, cudaStream_t stream) const;
    void launchExtend(const KernelParams& params, uint32_t launchCount, cudaStream_t stream) const;
    void launchConnect(const KernelParams& params, uint32_t launchCount, cudaStream_t stream) const;
    void launchGenerateAov(const KernelParams& params, cudaStream_t stream) const;
    void launchExtendAov(const KernelParams& params, cudaStream_t stream) const;
    void launchShadeAov(const KernelParams& params, cudaStream_t stream) const;
};
