#pragma once

#include <cstdint>
#include <memory>

#include <gpu/gpu.hpp>

#include "Shared/Camera.h"
#include "Shared/Environment.h"
#include "Shared/Frame.h"
#include "Shared/Raytracing.h"
#include "Mesh/Assets/Gaussian.h"
#include "Shared/Light.h"
#include "Shared/Lens.h"
#include "Optics/KolbLens.h"
#include <vector>

class Scene;
class RenderSettings;
// The CPU mesh asset, distinct from the nr::graphics::Mesh record.
class Mesh;

namespace gpu { class Device; }

// graphics API ray tracer and sole render owner. It holds the TLAS, the per-frame
// record, and the pointer tables that address resources the Scene owns.
class Raytracer
{
public:
    // The device is owned by the session, not by the renderer: the viewport
    // composite, the swapchain and this renderer all share one gpu::Device.
    Raytracer(gpu::Device& device, uint32_t width, uint32_t height,
        bool exportColorMemory = false);
    ~Raytracer();

    Raytracer(const Raytracer&) = delete;
    Raytracer& operator=(const Raytracer&) = delete;

    void resize(uint32_t width, uint32_t height);
    nr::graphics::Frame data{};
    gpu::Shared<nr::graphics::Lens> lens;
    // Publish changed resident records before recording a frame.
    void commit();
    // Rebuilds immutable graphics API BLAS/TLAS state from the scene's host geometry
    // mirror. The replacement is published only after all builds complete.
    void uploadScene(Scene& scene);
    // Applies fixed-topology transforms and Gaussian value edits in place.
    // Returns false when a structural rebuild is required.
    bool updateScene(const Scene& scene, bool updateGaussians);
    // Republishes every RenderSettings-derived push value. Settings such as
    // the Gaussian shading mode, proxy-overdraw counters, and transparent
    // background change no GPU resource, so hosts call this on its own rather
    // than forcing a scene re-upload just to make an edit observable.
    void applyRenderSettings(const RenderSettings& settings);
    void updateLights(const Scene& scene);
    void updateCamera(const Scene& scene);
    // Publishes the scene environment (colour, rotation, exposure, HDRI and
    // its importance CDF) as an immutable descriptor-heap record.
    void uploadEnvironment(Scene& scene);
    // Rebuilds the material pointer table. Only needed when the set of
    // materials changes; an edit to one material re-uploads only itself.
    void uploadMaterials(Scene& scene);
    // Dispatch one sample into the renderer-owned output texture. The work is
    // recorded into the enclosing gpu::Frame when the caller has one open,
    // and submitted on its own when it does not.
    void render(uint32_t frameIndex = 0, uint32_t sampleIndex = 0);
    // Valid after the command buffer containing the most recent record() has
    // completed. Returns actual device timestamp time, not CPU wall time.
    double lastDispatchMilliseconds();
    // The ray tracer's contract is a texture, not a window or swapchain. The
    // image handle is for native GPU operations/interop; the texture handle is
    // for sampling or storage access in another gpu pipeline. Both refer to
    // the same scene-linear RGBA32F image and remain valid until resize().
    gpu::ImageHandle outputImageHandle() const { return colorImage.handle(); }
    // Storage view for gpu shader pipelines that read/write the output.
    gpu::TextureHandle outputTexture() const { return colorImage.storage_handle(); }
    // Sampled view for gpu shader pipelines that only read the output.
    gpu::TextureHandle outputSampledTexture() const { return colorImage.sampled_handle(); }

    // Descriptor-heap handles for the AOV textures the viewport composite pass
    // reads. Heap handles the images already own, so consumers need no
    // descriptor allocation or renderer-specific presentation code.
    gpu::TextureHandle albedoTexture() const { return albedoImage.storage_handle(); }
    gpu::TextureHandle normalTexture() const { return normalImage.storage_handle(); }
    gpu::TextureHandle positionTexture() const { return positionImage.storage_handle(); }
    gpu::TextureHandle cryptomatteTexture() const { return cryptomatteImage.storage_handle(); }
    gpu::GpuPtr<std::uint32_t> gaussianOverdrawPtr() const { return gaussianOverdrawBuffer.ptr(); }
    gpu::Device& device() const { return *gpuDevice; }

    std::vector<std::byte> readColor();
    // Un-tonemapped scene-linear beauty.  Integrations writing an HDR image
    // (Hydra/F12 and Python) must use this rather than the 8-bit convenience
    // readback retained for legacy callers.
    std::vector<gpu::float4> readBeauty();
    std::vector<std::uint32_t> readCryptomatte();
    std::vector<gpu::float4> readPosition();
    uint32_t width() const { return renderWidth; }
    uint32_t height() const { return renderHeight; }

private:
    void createPipeline();
    void createImages();
    void updateRoot();
    void uploadLights(const Scene& scene);
    void uploadTextures(Scene& scene);

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
    // One device pointer per scene material, pointing at that material's own
    // nr::graphics::Material record.
    gpu::Buffer<std::uint64_t> materials;
    // The TLAS and the per-mesh/per-instance record tables. These used to be a
    // separate GpuScene object, but after resources became self-owning all it
    // held was a TLAS plus pointer tables into Scene's deques.
    struct GaussianProxy
    {
        gpu::Buffer<gpu::float3> positions;
        gpu::Buffer<std::uint32_t> indices;
        gpu::AccelerationStructure blas;
    };

    GaussianProxy gaussianProxy;
    std::vector<gpu::float4x4> gaussianTransforms;
    std::vector<gpu::Instance> tlasInstances;
    std::vector<uint32_t> meshInstanceAssetIndices;
    gpu::AccelerationStructure tlas;
    uint32_t instanceCount_{};
    // Host copy plus device allocation for each record table. The incremental
    // editor path rewrites the host copy and re-uploads the whole table; a
    // coalesced dirty range would just be a dirty queue by another name.
    // Addresses of each Mesh's own shared record, not the records themselves.
    std::vector<std::uint64_t> meshRecordData_;
    std::vector<nr::graphics::Instance> instanceData_;
    std::vector<Gaussian> gaussianRecordData_;
    std::vector<float> gaussianOpacityData_;
    std::vector<half> gaussianShCoefficientData_;
    gpu::Buffer<std::uint64_t> meshRecords_;
    gpu::Buffer<nr::graphics::Instance> instances_;
    gpu::Buffer<Gaussian> gaussianRecords_;
    gpu::Buffer<float> gaussianOpacities_;
    gpu::Buffer<half> gaussianShCoefficients_;
    gpu::Buffer<uint32_t> gaussianInstanceOffsets_;
    std::vector<const ::Mesh*> sourceMeshes_;
    std::vector<const GaussianAsset*> gaussianAssets_;
    std::vector<std::size_t> gaussianAssetCounts_;
    uint32_t gaussianCount_{};
    uint32_t gaussianShCoefficientCount_{};
    uint32_t gaussianProxyTriangleCount_{8u};
    uint32_t meshInstanceCount_{};
    uint32_t gaussianInstanceCount_{};
    uint32_t gaussianProxyType_{};
    float gaussianCutoffSigma_{};

    nr::graphics::Scene sceneBuffers() const;
    void uploadMeshRecords();
    void uploadInstances();
    void buildScene(Scene& scene);
    uint32_t buildMesh(const ::Mesh& asset);
    void buildGaussians(const Scene& scene);
    void buildTopLevel(const Scene& scene);
    void buildSceneData(const Scene& scene);
    // Updates fixed-topology editor mutations without replacing descriptors,
    // geometry BLASes, or the TLAS allocation. Returns false when topology or
    // proxy configuration changed and a transactional rebuild is required.
    bool updateMutableData(const Scene& scene, bool updateGaussians);
    gpu::Buffer<nr::graphics::PointLight> pointLights;
    gpu::Buffer<nr::graphics::SpotLight> spotLights;
    gpu::Buffer<nr::graphics::RectLight> rectLights;
    gpu::Buffer<nr::graphics::DirectionalLight> directionalLights;
    gpu::Buffer<nr::graphics::MeshLight> meshLights;
    gpu::Buffer<std::byte> energyLutBuffer;
    gpu::Buffer<std::byte> spectralTablesBuffer;
    // Sampled by any material whose texture failed to load at import time.
    gpu::Image<std::byte> whiteTexture;
};
