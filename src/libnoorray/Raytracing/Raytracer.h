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
#include "CUDA/Unique/Texture.h"
#include "CUDA/Unique/SharedImage.h"
#include "CUDA/Unique/DeviceBuffer.h"
#include "CUDA/Unique/Event.h"
#include "CUDA/Unique/ExternalSemaphore.h"
#include "CUDA/Unique/OptixModule.h"
#include "CUDA/Unique/OptixPipeline.h"
#include "CUDA/Unique/OptixProgramGroup.h"
#include "IO/Bitmap.h"
#include "Raytracing/SceneData.h"
#include "Raytracing/Tlas.h"

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
    uint32_t accumulatedSampleOffset{};
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
    void renderOfflineToDevice(float* rgbaDevice, uint32_t sampleCount);

    void setAovEnabled(bool enabled) { aovEnabled = enabled; }
    bool getAovEnabled() const { return aovEnabled; }
    void setStatsEnabled(bool enabled) { kernelStats.setEnabled(enabled); }
    void setTimingEnabled(bool enabled) { m_timingEnabled = enabled; }
    void harvestKernelStats() { kernelStats.harvestFrame(); }
    void printKernelStats() const { kernelStats.printReport(); }
    void updateMeshes();
    void updateTextures();
    void updateEnvironmentCdf();
    void updateLights();
    void updateTLAS();

    FrameInfo getFrameInfo() const;
    bool isRenderInFlight() const;
    bool isFrameReady() const;
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
    nr::cuda::UniqueSharedImage color[2];
    nr::cuda::UniqueSharedImage albedo[2];
    nr::cuda::UniqueSharedImage normal[2];
    nr::cuda::UniqueSharedImage cryptomatte[2];
    nr::cuda::UniqueSharedImage position[2];
    vk::UniqueSemaphore renderReady;
    vk::UniqueSemaphore bufferReleased;
    nr::cuda::UniqueExternalSemaphore cudaRenderReady;
    nr::cuda::UniqueExternalSemaphore cudaBufferReleased;
    Tlas tlas;
    WavefrontQueues queues{};
    nr::cuda::UniqueAsyncDeviceBuffer rayCountBuffer;
    nr::cuda::UniqueAsyncDeviceBuffer pathStateBuffer;
    nr::cuda::UniqueAsyncDeviceBuffer rayQueueBuffers[2];
    nr::cuda::UniqueAsyncDeviceBuffer hitQueueBuffer;
    nr::cuda::UniqueAsyncDeviceBuffer shadowQueueBuffer;
    nr::cuda::UniqueAsyncDeviceBuffer aovRayQueueBuffer;
    nr::cuda::UniqueAsyncDeviceBuffer aovHitQueueBuffer;
    nr::cuda::UniqueAsyncDeviceBuffer accumulationBuffer;
    nr::cuda::UniqueDeviceBuffer spectrumTableScaleDevice;
    nr::cuda::UniqueDeviceBuffer spectrumTableCoeffsDevice;
    nr::cuda::UniqueDeviceBuffer d65Device;
    nr::cuda::UniqueDeviceBuffer cieXDevice;
    nr::cuda::UniqueDeviceBuffer cieYDevice;
    nr::cuda::UniqueDeviceBuffer cieZDevice;
    nr::openpbr::EnergyLutStorage openPbrLutStorage;
    GpuSceneData gpuScene{};
    uint32_t nextBuffer{};
    uint32_t lastLaunched{};
    uint32_t aovStaleBuffers{2};
    uint64_t lastReadyValue{};
    uint64_t submittedFrame{};
    std::array<uint64_t, 2> lastUseValue{};
    nr::cuda::UniqueEvent m_startEvent;
    nr::cuda::UniqueEvent m_stopEvent;
    float m_gpuTimeMs = 0.0f;
    bool m_timingEnabled = false;
    bool m_eventsRecorded = false;
    KernelStats kernelStats;
    nr::cuda::UniqueDeviceBuffer optixLaunchParamsDevice;

    // OptiX state
    OptixDeviceContext optixCtx{};
    nr::cuda::UniqueOptixModule optixModule;
    nr::cuda::UniqueOptixProgramGroup optixExtendGroup;
    nr::cuda::UniqueOptixProgramGroup optixConnectGroup;
    nr::cuda::UniqueOptixProgramGroup optixProxyOverdrawGroup;
    nr::cuda::UniqueOptixProgramGroup optixTriangleGroup;
    nr::cuda::UniqueOptixProgramGroup optixGaussianHitGroup;
    nr::cuda::UniqueOptixProgramGroup optixProxyOverdrawHitGroup;
    nr::cuda::UniqueOptixProgramGroup optixMissGroup;
    nr::cuda::UniqueOptixPipeline optixPipeline;
    nr::cuda::UniqueOptixPipeline optixProxyOverdrawPipeline;
    nr::cuda::UniqueDeviceBuffer optixExtendRecord;
    nr::cuda::UniqueDeviceBuffer optixConnectRecord;
    nr::cuda::UniqueDeviceBuffer optixProxyOverdrawRecord;
    nr::cuda::UniqueDeviceBuffer optixHitgroupRecord;
    nr::cuda::UniqueDeviceBuffer optixProxyOverdrawHitgroupRecord;
    nr::cuda::UniqueDeviceBuffer optixMissRecord;
    OptixShaderBindingTable optixExtendSbt{};
    OptixShaderBindingTable optixConnectSbt{};
    OptixShaderBindingTable optixProxyOverdrawSbt{};

    void allocateQueues();
    void freeQueues() noexcept;
    void freeSceneData() noexcept;
    void launchGenerate(const KernelParams& params, cudaStream_t stream) const;
    void launchFinalize(const KernelParams& params, cudaStream_t stream) const;
    void launchResolveScatterPsf(const KernelParams& params, cudaStream_t stream) const;
    void launchApplyGatherPsf(const KernelParams& params, cudaStream_t stream) const;
    void prepareSensorFrame(Sensor& sensor, KernelParams& params, bool resetAccumulation);
    void launchSensorAddSample(const Sensor& sensor, const KernelParams& params, cudaStream_t stream);
    void applySensorAfterFrame(const Sensor& sensor, const KernelParams& params, cudaStream_t stream,
        bool finalSample);
    void launchShade(const KernelParams& params, uint32_t launchCount, cudaStream_t stream) const;
    void launchExtend(const KernelParams& params, uint32_t launchCount, cudaStream_t stream) const;
    void launchConnect(const KernelParams& params, uint32_t launchCount, cudaStream_t stream) const;
    void launchProxyOverdraw(const KernelParams& params, cudaStream_t stream) const;
    void launchGenerateAov(const KernelParams& params, cudaStream_t stream) const;
    void launchExtendAov(const KernelParams& params, cudaStream_t stream) const;
    void launchShadeAov(const KernelParams& params, cudaStream_t stream) const;
};
