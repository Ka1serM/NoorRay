#pragma once

#include <cstdint>
#include <memory>

#include <gpu/gpu.hpp>

#include "Backend/Vulkan/Raytracer/FrameAbi.h"
#include "Backend/Vulkan/Raytracer/CameraSnapshot.h"
#include "Backend/Vulkan/Raytracer/LightSnapshot.h"
#include "Rendering/Optics/KolbLens.h"
#include <vector>

class TriangleScene;
class VulkanScene;
class Scene;
class RenderSettings;

namespace gpu { class Device; }

// Vulkan-native ray tracer and sole render owner. It exercises the production
// ABI: a root pointer passed through gpu::Device, descriptor-heap resources,
// and a real ray-generation/miss/closest-hit pipeline.
class VulkanRaytracer
{
public:
    // The device is owned by the session, not by the renderer: the viewport
    // composite, the swapchain and this renderer all share one gpu::Device.
    VulkanRaytracer(gpu::Device& device, uint32_t width, uint32_t height,
        bool buildSmokeScene = false, bool exportColorMemory = false);
    ~VulkanRaytracer();

    VulkanRaytracer(const VulkanRaytracer&) = delete;
    VulkanRaytracer& operator=(const VulkanRaytracer&) = delete;

    void resize(uint32_t width, uint32_t height);
    // Upload an immutable native lens snapshot into a BDA storage buffer and
    // publish its address in the frame ABI. The renderer swaps the buffer only
    // between dispatches, so a frame never observes a partially rebuilt lens.
    void uploadLensSnapshot(const nr::optics::LensSnapshot& lens);
    // Publishes the active camera as an immutable descriptor-heap record.
    // cameraAddress in FrameParams carries this record's heap index for
    // compatibility with the existing pointer-width ABI.
    void uploadCameraSnapshot(const VulkanCameraSnapshot& camera);
    // Rebuilds immutable Vulkan BLAS/TLAS state from the scene's host geometry
    // mirror. The replacement is published only after all builds complete.
    void uploadScene(const Scene& scene);
    // Applies fixed-topology transforms and Gaussian value edits in place.
    // Returns false when a structural rebuild is required.
    bool updateScene(const Scene& scene, bool updateGaussians);
    // Republishes every RenderSettings-derived push value. Settings such as
    // the Gaussian shading mode, proxy-overdraw counters, and transparent
    // background change no GPU resource, so hosts call this on its own rather
    // than forcing a scene re-upload just to make an edit observable.
    void applyRenderSettings(const RenderSettings& settings);
    void updateLights(const Scene& scene);
    // Publishes the scene environment (colour, rotation, exposure, HDRI and
    // its importance CDF) as an immutable descriptor-heap record.
    void uploadEnvironment(const Scene& scene);
    void uploadMaterials(const Scene& scene,
        const std::vector<uint32_t>& svmWords,
        const std::vector<uint32_t>& svmTextureIndices);
    // Dispatch one sample. Recorded into the enclosing gpu::Frame when the
    // caller has one open, and submitted on its own when it does not - which
    // is what the offline renderer relies on.
    void render(uint32_t frameIndex = 0, uint32_t sampleIndex = 0);
    // Valid after the command buffer containing the most recent record() has
    // completed. Returns actual device timestamp time, not CPU wall time.
    double lastDispatchMilliseconds();
    // Composite the rendered beauty image onto another image - typically the
    // frame's presentation target. Extents may differ; the copy scales.
    void copyColorTo(gpu::ImageHandle target);

    // Descriptor-heap handles for the AOVs the viewport composite pass reads.
    // These are the slots gpu::Image already owns, so the viewport binds them
    // without allocating or writing any descriptor of its own.
    gpu::ImageHandle colorHandle() const { return colorImage.storage_handle(); }
    gpu::ImageHandle albedoHandle() const { return albedoImage.storage_handle(); }
    gpu::ImageHandle normalHandle() const { return normalImage.storage_handle(); }
    gpu::ImageHandle positionHandle() const { return positionImage.storage_handle(); }
    gpu::ImageHandle cryptomatteHandle() const { return cryptomatteImage.storage_handle(); }
    gpu::ResourceHandle gaussianOverdrawHandle() const { return gaussianOverdrawBuffer.handle(); }
    gpu::Device& device() const { return *gpuDevice; }

    std::vector<std::byte> readColor();
    // Un-tonemapped scene-linear beauty.  Integrations writing an HDR image
    // (Hydra/F12 and Python) must use this rather than the 8-bit convenience
    // readback retained for legacy callers.
    std::vector<gpu::float4> readBeauty();
    std::vector<std::uint32_t> readCryptomatte();
    std::vector<gpu::float4> readPosition();
    const nr::vulkan::FrameParams& frameParams() const { return params; }
    uint32_t width() const { return renderWidth; }
    uint32_t height() const { return renderHeight; }

private:
    void createPipeline();
    void createImages();
    void updateDescriptors();
    void uploadLights(const Scene& scene);

    gpu::Device* gpuDevice{};
    bool exportColorMemory{};
    uint32_t renderWidth{};
    uint32_t renderHeight{};
    gpu::Image<std::byte> colorImage;
    gpu::Image<std::byte> albedoImage;
    gpu::Image<std::byte> normalImage;
    gpu::Image<std::byte> positionImage;
    gpu::Image<std::byte> cryptomatteImage;
    gpu::Buffer<std::uint32_t> gaussianOverdrawBuffer;
    gpu::Buffer<gpu::float4> accumulationBuffer;
    gpu::Shader raygenShader;
    gpu::Shader missShader;
    gpu::Shader hitShader;
    gpu::Shader emissionHitShader;
    gpu::Shader opacityAnyHitShader;
    gpu::Shader gaussianAnyHitShader;
    gpu::Shader gaussianHitShader;
    gpu::RayTracingPipeline pipeline;
    gpu::TimestampQuery dispatchTimestamp{};
    nr::vulkan::FrameParams params{};
    gpu::Buffer<std::byte> lensBuffer;
    gpu::Buffer<std::byte> cameraBuffer;
    gpu::Buffer<std::byte> materialBuffer;
    gpu::Buffer<std::byte> svmWordsBuffer;
    std::unique_ptr<TriangleScene> triangleScene;
    std::unique_ptr<VulkanScene> nativeScene;
    gpu::Buffer<std::byte> lightsBuffer;
    gpu::Buffer<std::byte> environmentBuffer;
    gpu::Buffer<std::byte> energyLutBuffer;
    gpu::Buffer<std::byte> spectralTablesBuffer;
    gpu::Image<std::byte> environmentImage;
    gpu::Image<std::byte> environmentCdfImage;
    int uploadedEnvironmentTextureIndex{-2};
    // Entry zero is always a 1x1 white fallback. SVM texture-index words
    // contain descriptor-heap indices, so no additional FrameParams field is
    // needed for the texture table.
    std::vector<gpu::Image<std::byte>> textureImages;
};
