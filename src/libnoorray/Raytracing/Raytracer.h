#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include <noorrhi/noorrhi.hpp>

#include "Shared/Camera.h"
#include "Shared/Environment.h"
#include "Shared/Frame.h"
#include "Shared/Raytracing.h"
#include "Mesh/Assets/Gaussian.h"
#include "Shared/Light.h"
#include "Shared/Lens.h"
#include "Shared/RenderSettings.h"
#include "Optics/KolbLens.h"
#include <vector>

class Scene;
// The CPU mesh asset, distinct from the nr::graphics::Mesh record.
class Mesh;

namespace noorrhi { class Device; }
namespace nr::materialx { struct MaterialShader; }

// graphics API ray tracer and sole render owner. It holds the TLAS, the per-frame
// record, and the pointer tables that address resources the Scene owns.
//
// Scene publication, output images and readback are shared; each renderer
// only supplies its ray-tracing pipeline. Every implementation reads the same
// nr::graphics::Frame and writes the same beauty and AOV images, so hosts and
// the viewport composite treat them interchangeably.
class Raytracer
{
public:
    // The device is owned by the session, not by the renderer: the viewport
    // composite, the swapchain and this renderer all share one noorrhi::Device.
    static std::unique_ptr<Raytracer> create(noorrhi::Device& device,
        uint32_t width, uint32_t height,
        bool exportColorMemory = false);
    virtual ~Raytracer();

    virtual RaytracerType type() const noexcept = 0;
    virtual bool supportsMeshLights() const noexcept { return true; }
    // Whether materials must be compiled to SVM programs for this renderer.
    virtual bool needsSvmPrograms() const noexcept { return true; }

    Raytracer(const Raytracer&) = delete;
    Raytracer& operator=(const Raytracer&) = delete;

    // Sets the logical render size. A size that fits the current image
    // allocation only changes the traced rectangle: no GPU wait and no image
    // replacement. Without a reservation the allocation tracks the request
    // exactly, so image handles change on every size change.
    void resize(uint32_t width, uint32_t height);
    // Keeps the output images allocated at least this large, so interactive
    // hosts can resize within it every frame. Reallocating here synchronizes.
    void reserve(uint32_t width, uint32_t height);
    nr::graphics::Frame data{};
    noorrhi::Shared<nr::graphics::Lens> lens;
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
    // Brings renderer-owned resources in line with the current settings and
    // size. Reallocating a resource means destroying one the GPU may still be
    // reading, so this waits for the device; hosts must therefore call it
    // outside any recorded frame, before render(). render() repeats the check
    // for hosts that do not (the offline path), where waiting mid-frame is
    // harmless because there is no frame open.
    virtual void prepareFrameResources() {}
    // Dispatch one sample into the renderer-owned output texture. The work is
    // recorded into the enclosing noorrhi::Frame when the caller has one open,
    // and submitted on its own when it does not.
    virtual void render(uint32_t frameIndex = 0, uint32_t sampleIndex = 0);
    // Valid after the command buffer containing the most recent record() has
    // completed. Returns actual device timestamp time, not CPU wall time.
    double lastDispatchMilliseconds();
    // The ray tracer's contract is a texture, not a window or swapchain. The
    // image handle is for native GPU operations/interop; the texture handle is
    // for sampling or storage access in another NoorRHI pipeline. Both refer to
    // the same scene-linear RGBA32F image and remain valid until resize() or
    // reserve() reallocates. The image may be larger than width() x height();
    // only the bottom-left logical rectangle holds the current render.
    noorrhi::ImageHandle outputImageHandle() const { return colorImage.handle(); }
    // Storage view for NoorRHI shader pipelines that read/write the output.
    noorrhi::TextureHandle outputTexture() const { return colorImage.storage_handle(); }
    // Sampled view for NoorRHI shader pipelines that only read the output.
    noorrhi::TextureHandle outputSampledTexture() const { return colorImage.sampled_handle(); }

    // Descriptor-heap handles for the AOV textures the viewport composite pass
    // reads. Heap handles the images already own, so consumers need no
    // descriptor allocation or renderer-specific presentation code.
    noorrhi::TextureHandle albedoTexture() const { return albedoImage.storage_handle(); }
    noorrhi::TextureHandle normalTexture() const { return normalImage.storage_handle(); }
    noorrhi::TextureHandle positionTexture() const { return positionImage.storage_handle(); }
    noorrhi::TextureHandle cryptomatteTexture() const { return cryptomatteImage.storage_handle(); }
    noorrhi::GpuPtr<std::uint32_t> gaussianOverdrawPtr() const { return gaussianOverdrawBuffer.ptr(); }
    noorrhi::Device& device() const { return *gpuDevice; }

    std::vector<std::byte> readColor();
    // Un-tonemapped scene-linear beauty.  Integrations writing an HDR image
    // (Hydra/F12 and Python) must use this rather than the 8-bit convenience
    // readback retained for legacy callers.
    std::vector<noorrhi::float4> readBeauty();
    std::vector<std::uint32_t> readCryptomatte();
    std::vector<noorrhi::float4> readPosition();
    // Single-texel reads for picking. (x, y) is a render pixel with the origin
    // at the bottom-left, inside width() x height(); only that texel is copied.
    // Out-of-range pixels read as a miss (~0u / zero).
    std::uint32_t readCryptomatteAt(uint32_t x, uint32_t y);
    noorrhi::float4 readPositionAt(uint32_t x, uint32_t y);
    uint32_t width() const { return renderWidth; }
    uint32_t height() const { return renderHeight; }
    // The resolution rays are actually traced at. A renderer that upscales -
    // the realtime path, through FSR - traces a fixed, smaller resolution and
    // upscales to the logical size; by default the two are the same.
    virtual uint32_t traceWidth() const { return renderWidth; }
    virtual uint32_t traceHeight() const { return renderHeight; }
    // Allocated size of every output image and per-pixel buffer.
    uint32_t imageWidth() const { return imageWidth_; }
    uint32_t imageHeight() const { return imageHeight_; }

protected:
    Raytracer(noorrhi::Device& device, uint32_t width, uint32_t height,
        bool exportColorMemory);

    // Concrete renderers compile and own their shader pipeline. The rest of
    // the renderer state is intentionally shared so scene publication,
    // accumulation, AOVs and viewport integration stay renderer-agnostic.
    noorrhi::RayTracingPipeline pipeline;
    virtual void renderImpl();
    virtual void onImageAllocationChanged() {}
    noorrhi::Device& renderDevice() const { return *gpuDevice; }
    uint32_t logicalRenderWidth() const { return renderWidth; }
    uint32_t logicalRenderHeight() const { return renderHeight; }
    // Called at the end of applyRenderSettings(), for settings only one
    // renderer reads.
    virtual void onRenderSettingsApplied(const RenderSettings&) {}
    // Called after every light upload, with the records the GPU now holds.
    virtual void onLightsUploaded() {}
    // Called when material uploads added callable shaders. The list only
    // grows, so a material's index into it stays valid.
    virtual void onMaterialShadersChanged(std::span<const noorrhi::Shader>) {}
    const std::vector<nr::graphics::PointLight>& pointLightRecords() const { return pointLightData_; }
    const std::vector<nr::graphics::SpotLight>& spotLightRecords() const { return spotLightData_; }
    const std::vector<nr::graphics::RectLight>& rectLightRecords() const { return rectLightData_; }
    const std::vector<nr::graphics::DirectionalLight>& directionalLightRecords() const
    {
        return directionalLightData_;
    }

private:
    void createImages();
    void updateRoot();
    void uploadLights(const Scene& scene);
    void uploadTextures(Scene& scene);

    noorrhi::Device* gpuDevice{};
    bool exportColorMemory{};
    uint32_t renderWidth{};
    uint32_t renderHeight{};
    uint32_t imageWidth_{};
    uint32_t imageHeight_{};
    uint32_t reservedWidth_{};
    uint32_t reservedHeight_{};
    void reallocate(uint32_t width, uint32_t height);
    template<class T>
    std::vector<T> cropToRender(std::vector<T> pixels) const;
    noorrhi::Image<std::byte> colorImage;
    noorrhi::Image<std::byte> albedoImage;
    noorrhi::Image<std::byte> normalImage;
    noorrhi::Image<std::byte> positionImage;
    noorrhi::Image<std::byte> cryptomatteImage;
    noorrhi::Buffer<std::uint32_t> gaussianOverdrawBuffer;
    noorrhi::Buffer<noorrhi::float4> accumulationBuffer;
    noorrhi::TimestampQuery dispatchTimestamp{};
    // One device pointer per scene material, pointing at that material's own
    // nr::graphics::Material record.
    noorrhi::Buffer<std::uint64_t> materials;
    // Callable shaders of the materials, in shader-binding-table order, and
    // the compiled programs they were created from.
    std::vector<std::shared_ptr<const nr::materialx::MaterialShader>> materialShaderPrograms_;
    std::vector<noorrhi::Shader> materialShaders_;
    // The TLAS and the per-mesh/per-instance record tables. These used to be a
    // separate GpuScene object, but after resources became self-owning all it
    // held was a TLAS plus pointer tables into Scene's deques.
    struct GaussianProxy
    {
        noorrhi::Buffer<noorrhi::float3> positions;
        noorrhi::Buffer<std::uint32_t> indices;
        noorrhi::AccelerationStructure blas;
    };

    GaussianProxy gaussianProxy;
    std::vector<noorrhi::float4x4> gaussianTransforms;
    std::vector<noorrhi::Instance> tlasInstances;
    std::vector<uint32_t> meshInstanceAssetIndices;
    noorrhi::AccelerationStructure tlas;
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
    noorrhi::Buffer<std::uint64_t> meshRecords_;
    noorrhi::Buffer<nr::graphics::Instance> instances_;
    noorrhi::Buffer<Gaussian> gaussianRecords_;
    noorrhi::Buffer<float> gaussianOpacities_;
    noorrhi::Buffer<half> gaussianShCoefficients_;
    noorrhi::Buffer<uint32_t> gaussianInstanceOffsets_;
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
    noorrhi::Buffer<nr::graphics::PointLight> pointLights;
    noorrhi::Buffer<nr::graphics::SpotLight> spotLights;
    noorrhi::Buffer<nr::graphics::RectLight> rectLights;
    noorrhi::Buffer<nr::graphics::DirectionalLight> directionalLights;
    noorrhi::Buffer<nr::graphics::MeshLight> meshLights;
    // Host copies of the typed light buffers. An edit that keeps every count,
    // such as moving a light, rewrites these and uploads only the span of
    // records that changed into the existing buffers - the same way mesh
    // instance transforms are updated - instead of reallocating all five.
    std::vector<nr::graphics::PointLight> pointLightData_;
    std::vector<nr::graphics::SpotLight> spotLightData_;
    std::vector<nr::graphics::RectLight> rectLightData_;
    std::vector<nr::graphics::DirectionalLight> directionalLightData_;
    std::vector<nr::graphics::MeshLight> meshLightData_;
    bool lightBuffersValid_{};
    noorrhi::Buffer<std::byte> energyLutBuffer;
    noorrhi::Buffer<std::byte> spectralTablesBuffer;
    // Sampled by any material whose texture failed to load at import time.
    noorrhi::Image<std::byte> whiteTexture;
};
