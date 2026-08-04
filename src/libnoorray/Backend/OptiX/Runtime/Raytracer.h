#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>
#include <glm/vec4.hpp>
#include <optix.h>
#include <vulkan/vulkan.hpp>

#include "Backend/CUDA/KernelStats.h"
#include "Backend/CUDA/Unique/Texture.h"
#include "Backend/CUDA/Unique/SharedImage.h"
#include "Backend/CUDA/Unique/SharedBuffer.h"
#include "Backend/CUDA/Unique/DeviceBuffer.h"
#include "Backend/CUDA/Unique/Event.h"
#include "Backend/CUDA/Unique/ExternalSemaphore.h"
#include "Backend/CUDA/Unique/OptixModule.h"
#include "Backend/CUDA/Unique/OptixPipeline.h"
#include "Backend/CUDA/Unique/OptixProgramGroup.h"
#include "Backend/OptiX/ABI/SceneData.h"
#include "Backend/OptiX/ABI/GpuSceneCache.h"
#include "Backend/OptiX/Denoising/OptixDenoiser.h"
#include "Backend/OptiX/Acceleration/Tlas.h"
#include "Backend/OptiX/Acceleration/AnalyticLightBlas.h"
#include "Materials/Shading/EnergyLut.h"
#include "Materials/SVM/SvmProgramTable.h"

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

    cudaStream_t getCudaStream() const { return stream; }

    InteropFrame getInteropFrame() const;
    bool isRenderInFlight() const;
    void waitForRender() const;
    bool isFrameReady() const;
    float getGpuTimeMs();
    uint32_t getWidth() const { return width; }
    uint32_t getHeight() const { return height; }
    uint32_t getScratchCapacity() const { return scratchCapacity; }
    void debugSave(const std::string& path) const;
    Image& getOutputColor() { return color.getImage(); }
    Image& getOutputAlbedo() { return albedo.getImage(); }
    Image& getOutputNormal() { return normal.getImage(); }
    Image& getOutputCrypto() { return cryptomatte.getImage(); }
    Image& getOutputPosition() { return position.getImage(); }
    const nr::cuda::UniqueSharedBuffer& getOutputDenoised() const { return denoised; }
    nr::cuda::UniqueSharedBuffer& getOutputDenoised() { return denoised; }
    bool hasDenoisedOutput() const { return denoisedOutputAvailable; }

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
    nr::cuda::UniqueSharedBuffer denoised;
    bool denoisedOutputAvailable{};
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
    bool denoiserGuidesStale{true};
    nr::cuda::UniqueDeviceBuffer spectrumTableScaleDevice;
    nr::cuda::UniqueDeviceBuffer spectrumTableCoeffsDevice;
    nr::cuda::UniqueTexture spectrumTableTexture;
    nr::shading::energy_lut::Storage energyLutStorage;
    nr::cuda::UniqueDeviceBuffer d65Device;
    nr::cuda::UniqueDeviceBuffer cieXDevice;
    nr::cuda::UniqueDeviceBuffer cieYDevice;
    nr::cuda::UniqueDeviceBuffer cieZDevice;
    nr::cuda::UniqueDeviceBuffer lightAliasDevice;
    nr::cuda::UniqueDeviceBuffer lightTreeDevice;
    nr::cuda::UniqueDeviceBuffer directLightCandidateDevice;
    nr::cuda::UniqueDeviceBuffer meshLightCandidateOffsetsDevice;
    nr::cuda::UniqueDeviceBuffer meshLightCandidateIndicesDevice;
    nr::cuda::UniqueDeviceBuffer analyticLightBvhCandidateDevice;
    nr::cuda::UniqueDeviceBuffer meshLightBvhCandidateDevice;
    AnalyticLightBlas analyticLightBlas;
    AnalyticLightBlas meshLightBlas;
    GpuSceneCache gpuCache;
    uint64_t lastReadyValue{};
    uint64_t submittedFrame{};
    nr::cuda::UniqueEvent m_startEvent;
    nr::cuda::UniqueEvent m_stopEvent;
    nr::cuda::UniqueEvent m_frameCompleteEvent;
    float m_gpuTimeMs = 0.0f;
    bool m_timingEnabled = false;
    bool m_eventsRecorded = false;
    bool m_frameCompleteRecorded = false;
    KernelStats kernelStats;
    nr::OptixDenoiser denoiser;
    nr::cuda::UniqueDeviceBuffer optixLaunchParamsDevice;

    // OptiX state
    OptixDeviceContext optixCtx{};
    nr::cuda::UniqueOptixModule optixModule;
    nr::cuda::UniqueOptixModule optixPathMissModule;
    nr::cuda::UniqueOptixProgramGroup optixPathTraceGroup;
    nr::cuda::UniqueOptixProgramGroup optixProxyOverdrawGroup;
    nr::cuda::UniqueOptixProgramGroup optixPathMeshHitGroup;
    nr::cuda::UniqueOptixProgramGroup optixPathGaussianHitGroup;
    nr::cuda::UniqueOptixProgramGroup optixTriangleGroup;
    nr::cuda::UniqueOptixProgramGroup optixGaussianHitGroup;
    nr::cuda::UniqueOptixProgramGroup optixAnalyticLightHitGroup;
    nr::cuda::UniqueOptixProgramGroup optixProxyOverdrawHitGroup;
    nr::cuda::UniqueOptixProgramGroup optixMissGroup;
    nr::cuda::UniqueOptixProgramGroup optixPathMissGroup;

    nr::svm::SvmProgramTable materialxPrograms;
    nr::cuda::UniqueDeviceBuffer svmWordsDevice;
    nr::cuda::UniqueDeviceBuffer svmTextureIndicesDevice;

    // Beauty and traversal-only query records share one SBT. Query rays use
    // QuerySbtRecordOffset to select the appended traversal-only records.
    nr::cuda::UniqueDeviceBuffer pathHitgroupRecordBase;
    nr::cuda::UniqueOptixPipeline optixPipeline;
    nr::cuda::UniqueOptixPipeline optixProxyOverdrawPipeline;
    nr::cuda::UniqueDeviceBuffer optixPathTraceRecord;
    nr::cuda::UniqueDeviceBuffer optixProxyOverdrawRecord;
    nr::cuda::UniqueDeviceBuffer optixProxyOverdrawHitgroupRecord;
    nr::cuda::UniqueDeviceBuffer optixMissRecord;
    nr::cuda::UniqueDeviceBuffer optixPathMissRecord;
    OptixShaderBindingTable optixPathTraceSbt{};
    OptixShaderBindingTable optixProxyOverdrawSbt{};

    void allocateScratchBuffers();
    void freeScratchBuffers() noexcept;
    void freeSceneData() noexcept;
    void rebuildPipeline();
    nr::cuda::UniqueOptixPipeline prepareSvmPipeline() const;
    void installSvmPipeline(nr::cuda::UniqueOptixPipeline pipeline);
    void rebuildMeshHitgroupSbt();
    void uploadSvmPrograms();
    OptixPipelineCompileOptions pipelineCompileOptions{};
    void launchPsfResolve(const KernelParams& params, cudaStream_t stream) const;
    void launchDenoiser(
        const KernelParams& params, cudaStream_t stream, bool useAovGuides);
    void prepareSensorFrame(Sensor& sensor, KernelParams& params, bool resetAccumulation);
    void ensureProxyOverdrawResources();
    void ensureSemaphores();
    void ensureAovImages();
    void releaseAovImages() noexcept;
    void launchPathTrace(const KernelParams& params, cudaStream_t stream,
        bool uploadStaticParams) const;
    void launchProxyOverdraw(const KernelParams& params, cudaStream_t stream) const;
};
