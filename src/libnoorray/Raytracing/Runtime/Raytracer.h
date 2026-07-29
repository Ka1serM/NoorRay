#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>
#include <glm/vec4.hpp>
#include <optix.h>
#include <vulkan/vulkan.hpp>

#include "CUDA/KernelStats.h"
#include "Training/GaussianTrainData.h"
#include "CUDA/Unique/Texture.h"
#include "CUDA/Unique/SharedImage.h"
#include "CUDA/Unique/DeviceBuffer.h"
#include "CUDA/Unique/Event.h"
#include "CUDA/Unique/ExternalSemaphore.h"
#include "CUDA/Unique/OptixModule.h"
#include "CUDA/Unique/OptixPipeline.h"
#include "CUDA/Unique/OptixProgramGroup.h"
#include "Raytracing/Gpu/SceneData.h"
#include "Raytracing/Gpu/GpuSceneCache.h"
#include "Raytracing/Denoising/OptixDenoiserState.h"
#include "Raytracing/Acceleration/Tlas.h"
#include "Shading/EnergyLut.h"
#include "SVM/SvmProgramTable.h"

class CameraInstance;
class Context;
class Image;
class Scene;
class Sensor;

struct InteropFrame
{
    uint32_t bufferIndex{};
    uint64_t readyValue{};
    vk::Semaphore renderReadySemaphore{};
    vk::Semaphore bufferReleasedSemaphore{};
};

class Raytracer
{
public:
    Raytracer(Context& context, Scene& scene);
    ~Raytracer();

    Raytracer(const Raytracer&) = delete;
    Raytracer& operator=(const Raytracer&) = delete;

    void resize(uint32_t width, uint32_t height);
    void renderFrame(uint32_t frameIndex = 0, uint32_t accumulatedSamples = 0);

    nr::svm::SvmProgramRecord registerMaterialXProgram(const nr::svm::CompiledSvmProgram& program);
    nr::svm::SvmProgramRecord replaceMaterialXProgram(
        const nr::svm::CompiledSvmProgram& program);
    void releaseMaterialXProgram(uint32_t index);
    uint32_t getAvailableMaterialXProgramCount() const {
        return static_cast<uint32_t>(materialxPrograms.records().size());
    }
    void clearMaterialXPrograms();

    void setAovEnabled(bool enabled);
    bool getAovEnabled() const;
    bool hasAovImages() const { return aovImagesCreated; }
    void setStatsEnabled(bool enabled) { kernelStats.setEnabled(enabled); }
    void setTimingEnabled(bool enabled) { m_timingEnabled = enabled; }
    void harvestKernelStats() { kernelStats.harvestFrame(); }
    void printKernelStats() const { kernelStats.printReport(); }
    void updateMeshes();
    void updateTextures();
    void updateEnvironmentCdf();
    void updateLights();
    void updateTLAS();

    OptixTraversableHandle getGaussianTlasHandle() const { return tlas.getTraversable(); }
    OptixDeviceContext getOptixDeviceContext() const { return optixCtx; }
    cudaStream_t getCudaStream() const { return stream; }
    void renderGaussianTrainForward(const GaussianTrainParams& params,
        uint32_t width, uint32_t height);
    void renderGaussianTrainBackward(const GaussianTrainParams& params,
        uint32_t width, uint32_t height);

    InteropFrame getInteropFrame() const;
    bool isRenderInFlight() const;
    void waitForRender() const;
    bool isFrameReady() const;
    float getGpuTimeMs();
    float getAverageNoiseVariance() const;
    uint32_t getWidth() const { return width; }
    uint32_t getHeight() const { return height; }
    uint32_t getScratchCapacity() const { return scratchCapacity; }
    void debugSave(const std::string& path) const;
    Image& getOutputColor() { return color.getImage(); }
    Image& getOutputAlbedo() { return albedo.getImage(); }
    Image& getOutputNormal() { return normal.getImage(); }
    Image& getOutputCrypto() { return cryptomatte.getImage(); }
    Image& getOutputPosition() { return position.getImage(); }

private:
    static constexpr uint32_t MaxBounces = 66;

    Context& context;
    Scene& scene;
    bool aovEnabled{false};
    bool aovAvailable{};
    bool aovStale{true};
    uint32_t width{};
    uint32_t height{};
    cudaStream_t stream{};
    nr::cuda::UniqueSharedImage color;
    nr::cuda::UniqueSharedImage albedo;
    nr::cuda::UniqueSharedImage normal;
    nr::cuda::UniqueSharedImage cryptomatte;
    nr::cuda::UniqueSharedImage position;
    bool aovImagesCreated{};
    vk::UniqueSemaphore renderReady;
    vk::UniqueSemaphore bufferReleased;
    nr::cuda::UniqueExternalSemaphore cudaRenderReady;
    nr::cuda::UniqueExternalSemaphore cudaBufferReleased;
    Tlas tlas;
    uint32_t scratchCapacity{};
    nr::cuda::UniqueAsyncDeviceBuffer accumulationBuffer;
    nr::cuda::UniqueAsyncDeviceBuffer denoiserAlbedoGuideBuffer;
    nr::cuda::UniqueAsyncDeviceBuffer denoiserNormalGuideBuffer;
    nr::cuda::UniqueAsyncDeviceBuffer noiseMomentsBuffer;
    nr::cuda::UniqueAsyncDeviceBuffer noiseVarianceSumBuffer;
    nr::cuda::UniqueDeviceBuffer spectrumTableScaleDevice;
    nr::cuda::UniqueDeviceBuffer spectrumTableCoeffsDevice;
    nr::cuda::UniqueTexture spectrumTableTexture;
    nr::shading::energy_lut::Storage energyLutStorage;
    nr::cuda::UniqueDeviceBuffer d65Device;
    nr::cuda::UniqueDeviceBuffer cieXDevice;
    nr::cuda::UniqueDeviceBuffer cieYDevice;
    nr::cuda::UniqueDeviceBuffer cieZDevice;
    nr::cuda::UniqueDeviceBuffer analyticLightAliasDevice;
    GpuSceneCache gpuCache;
    uint64_t lastReadyValue{};
    uint64_t submittedFrame{};
    nr::cuda::UniqueEvent m_startEvent;
    nr::cuda::UniqueEvent m_stopEvent;
    float m_gpuTimeMs = 0.0f;
    bool m_timingEnabled = false;
    bool m_eventsRecorded = false;
    float* m_noiseVarianceSumHost{};
    uint32_t m_noiseResultSampleCount{};
    KernelStats kernelStats;
    OptixDenoiserState denoiser;
    nr::cuda::UniqueDeviceBuffer optixLaunchParamsDevice;

    // OptiX state
    OptixDeviceContext optixCtx{};
    nr::cuda::UniqueOptixModule optixModule;
    nr::cuda::UniqueOptixProgramGroup optixPathTraceGroup;
    nr::cuda::UniqueOptixProgramGroup optixAovGroup;
    nr::cuda::UniqueOptixProgramGroup optixProxyOverdrawGroup;
    nr::cuda::UniqueOptixProgramGroup optixTriangleGroup;
    nr::cuda::UniqueOptixProgramGroup optixGaussianHitGroup;
    nr::cuda::UniqueOptixProgramGroup optixProxyOverdrawHitGroup;
    nr::cuda::UniqueOptixProgramGroup optixMissGroup;

    nr::svm::SvmProgramTable materialxPrograms;
    nr::cuda::UniqueDeviceBuffer svmWordsDevice;
    nr::cuda::UniqueDeviceBuffer svmTextureIndicesDevice;

    // One combined buffer, shared by path-trace and AOV SBTs: one common mesh
    // hitgroup record followed by one Gaussian record. Per-face material
    // selection happens in Geometry.h and never changes this SBT.
    nr::cuda::UniqueDeviceBuffer meshHitgroupRecordBase;
    nr::cuda::UniqueOptixPipeline optixPipeline;
    nr::cuda::UniqueOptixPipeline optixProxyOverdrawPipeline;
    nr::cuda::UniqueOptixModule optixTrainingModule;
    nr::cuda::UniqueOptixProgramGroup optixTrainingExtendGroup;
    nr::cuda::UniqueOptixProgramGroup optixTrainingGaussianHitGroup;
    nr::cuda::UniqueOptixPipeline optixTrainingPipeline;
    nr::cuda::UniqueDeviceBuffer optixPathTraceRecord;
    nr::cuda::UniqueDeviceBuffer optixAovRecord;
    nr::cuda::UniqueDeviceBuffer optixProxyOverdrawRecord;
    nr::cuda::UniqueDeviceBuffer optixProxyOverdrawHitgroupRecord;
    nr::cuda::UniqueDeviceBuffer optixMissRecord;
    nr::cuda::UniqueDeviceBuffer optixTrainingExtendRecord;
    nr::cuda::UniqueDeviceBuffer optixTrainingHitgroupRecord;
    OptixShaderBindingTable optixPathTraceSbt{};
    OptixShaderBindingTable optixAovSbt{};
    OptixShaderBindingTable optixProxyOverdrawSbt{};
    OptixShaderBindingTable optixTrainingExtendSbt{};

    void allocateScratchBuffers();
    void freeScratchBuffers() noexcept;
    void freeSceneData() noexcept;
    void rebuildPipeline();
    nr::cuda::UniqueOptixPipeline prepareSvmPipeline() const;
    void installSvmPipeline(nr::cuda::UniqueOptixPipeline pipeline);
    void rebuildMeshHitgroupSbt();
    void uploadSvmPrograms();
    OptixPipelineCompileOptions pipelineCompileOptions{};
    static constexpr uint32_t PostProcessPsf = 1u << 0u;
    static constexpr uint32_t PostProcessNoiseVariance = 1u << 1u;

    void launchPostProcess(
        const KernelParams& params, cudaStream_t stream, uint32_t flags) const;
    void launchDenoiser(
        const KernelParams& params, cudaStream_t stream, bool useAovGuides);
    void prepareSensorFrame(Sensor& sensor, KernelParams& params, bool resetAccumulation);
    void ensureTrainingResources();
    void ensureProxyOverdrawResources();
    void ensureSemaphores();
    void ensureAovImages();
    void releaseAovImages() noexcept;
    void launchGaussianTrainPath(
        const GaussianTrainingKernelParams& params,
        uint32_t launchCount, cudaStream_t stream) const;
    void launchPathTrace(const KernelParams& params, cudaStream_t stream) const;
    void launchAov(const KernelParams& params, cudaStream_t stream) const;
    void launchProxyOverdraw(const KernelParams& params, cudaStream_t stream) const;
};
