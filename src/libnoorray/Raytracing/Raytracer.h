#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include <noorrhi/noorrhi.hpp>

#include "Shared/Camera.h"
#include "Shared/Environment.h"
#include "Shared/Frame.h"
#include "Shared/Raytracing.h"
#include "Shared/Light.h"
#include "Shared/Lens.h"
#include "Shared/RenderSettings.h"
#include "Optics/KolbLens.h"
#include "Mesh/Assets/Mesh.h"
#include "Mesh/SplineMeshPass.h"
#include "Scene/Scene.h"
// The CPU mesh asset, distinct from the nr::graphics::Mesh record.
class Mesh;
class MeshInstance;

namespace noorrhi { class Device; }
namespace nr::materialx { struct MaterialShader; }

// A compiled material's hit stages (MaterialHit.slang).
struct MaterialHitShaders
{
    noorrhi::Shader closestHit;
    noorrhi::Shader anyHit;
    noorrhi::Shader shadowAnyHit;
};

// The driver work between publishing a MaterialX program and using it for
// tracing. Hosts use this to distinguish the visible post-compilation pause.
enum class MaterialShaderStage { Idle, CompilingLibraries, LinkingPipeline };

// One shader-binding-table hit record of the shared TLAS, in record order: a
// mesh instance's hit offset addresses RaytracingRayTypeCount records per
// section of its mesh, drawn with its materials. A section drawn with a
// Gaussian splat material is a Gaussian record, which the splat hit stages
// serve.
struct HitRecord
{
    enum class Kind { Section, Gaussian };
    Kind kind{};
    uint32_t rayType{};
    // A section's material: the index of its hit shaders in the list
    // onMaterialShadersChanged() received, ~0u without compiled shaders.
    uint32_t materialShaders{~0u};
    // The material's opacity may be below one.
    bool transparent{};
    // Shadow rays skip the section although other sections of its mesh cast
    // shadows: its shadow any-hit stage ignores every hit.
    bool shadowFiltered{};
    bool operator==(const HitRecord&) const = default;
};

// Common scene and output-image resources, composed into each concrete
// renderer. This is deliberately not a base class: renderer-specific
// construction owns its resources and decides which common resources it needs.
class RaytracerResources
{
public:
    enum class FullOutputAovs { Written, Omitted };
    struct Callbacks
    {
        std::function<void()> imageAllocationChanged;
        std::function<void(const RenderSettings&)> renderSettingsApplied;
        std::function<void()> lightsUploaded;
        std::function<void(std::span<const MaterialHitShaders>)> materialShadersChanged;
        std::function<void(std::span<const HitRecord>)> hitRecordsChanged;
    };

    RaytracerResources(noorrhi::Device& device, uint32_t width, uint32_t height,
        bool exportColorMemory, FullOutputAovs fullOutputAovs, bool allocateAccumulationBuffer);
    ~RaytracerResources();
    RaytracerResources(const RaytracerResources&) = delete;
    RaytracerResources& operator=(const RaytracerResources&) = delete;
    void setCallbacks(Callbacks callbacks) { callbacks_ = std::move(callbacks); }

    // Sets the logical render size; the images follow in
    // prepareFrameResources(). Allocations are 150% of the size they are made
    // for, so a larger size reallocates to 150% of itself, and a growing
    // viewport reallocates a few times rather than on every step; the
    // allocation shrinks to 150% of a size once it has settled.
    void resize(uint32_t width, uint32_t height);
    nr::graphics::Frame data{};
    noorrhi::Shared<nr::graphics::Lens> lens;
    // Publish changed resident records before recording a frame.
    void commit();
    // Publishes the whole scene from scratch, dropping whatever the scene's
    // change lists hold. Waits for the device.
    void uploadScene(Scene& scene);
    // Publishes what the scene's change lists name since the last
    // publication: new textures, changed materials and meshes, and changed
    // instance slots. The cost follows the size of the change; only the TLAS
    // build or refit is proportional to the instance count.
    void publishScene(Scene& scene);
    // Republishes every RenderSettings-derived push value. Settings such as
    // the Gaussian shading mode, proxy-overdraw counters, and transparent
    // background change no GPU resource, so hosts call this on its own rather
    // than forcing a scene re-upload just to make an edit observable.
    void applyRenderSettings(const RenderSettings& settings);
    // Patches the light records `changed` names; a change in the light set
    // republishes them all.
    void updateLights(const Scene& scene, const Scene::LightIndices& changed);
    void updateCamera(const Scene& scene);
    // Publishes the scene environment (colour, rotation, exposure, HDRI and
    // its importance CDF) as an immutable descriptor-heap record.
    void uploadEnvironment(Scene& scene);
    // Brings renderer-owned resources in line with the current settings and
    // size, and releases image allocation a settled size no longer uses.
    // Reallocating a resource means destroying one the GPU may still be
    // reading, so this waits for the device; hosts must therefore call it
    // outside any recorded frame, before render(). render() repeats the check
    // for hosts that do not (the offline path), where waiting mid-frame is
    // harmless because there is no frame open. Returns true when the common
    // output images were replaced; the concrete renderer can rebuild its own
    // size-dependent resources in response.
    bool prepareFrameResources();
    // Records a concrete renderer's work with the common timestamp and frame
    // data setup. The callable records into the active frame, if any.
    void dispatch(uint32_t frameIndex, uint32_t sampleIndex, const std::function<void()>& record);
    // Valid after the command buffer containing the most recent record() has
    // completed. Returns actual device timestamp time, not CPU wall time.
    double lastDispatchMilliseconds();
    // The ray tracer's contract is a texture, not a window or swapchain. The
    // image handle is for native GPU operations/interop; the texture handle is
    // for sampling or storage access in another NoorRHI pipeline. Both refer to
    // the same scene-linear RGBA32F image and remain valid until a resize
    // reallocates. The image may be larger than width() x height();
    // only the bottom-left logical rectangle holds the current render.
    noorrhi::ImageHandle outputImageHandle() const { return colorImage.handle(); }
    // Storage view for NoorRHI shader pipelines that read/write the output.
    noorrhi::TextureHandle outputTexture() const { return colorImage.storage_handle(); }
    // Sampled view for NoorRHI shader pipelines that only read the output.
    noorrhi::TextureHandle outputSampledTexture() const { return colorImage.sampled_handle(); }

    // Descriptor-heap handles for the AOV textures the viewport composite pass
    // reads. Heap handles the images already own, so consumers need no
    // descriptor allocation or renderer-specific presentation code. Albedo,
    // normal and position are empty, with a zero handle, for a renderer that
    // does not write them (FullOutputAovs::Omitted).
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
    // Allocated size of every output image and per-pixel buffer.
    uint32_t imageWidth() const { return imageWidth_; }
    uint32_t imageHeight() const { return imageHeight_; }

    const std::vector<nr::graphics::PointLight>& pointLightRecords() const { return pointLightData_; }
    const std::vector<nr::graphics::SpotLight>& spotLightRecords() const { return spotLightData_; }
    const std::vector<nr::graphics::RectLight>& rectLightRecords() const { return rectLightData_; }
    const std::vector<nr::graphics::DirectionalLight>& directionalLightRecords() const
    { return directionalLightData_; }
private:
    // Hooks keep renderer-specific state synchronized with common publications.
    Callbacks callbacks_;
    void createImages();
    void updateRoot();
    void uploadLights(const Scene& scene);
    // Republishes what the light samplers derive from every selection weight.
    void publishLightSampling();
    void uploadTextures(Scene& scene);

    noorrhi::Device* gpuDevice{};
    SplineMeshPass splineMeshPass_;
    bool exportColorMemory{};
    bool allocateAccumulationBuffer{};
    uint32_t renderWidth{};
    uint32_t renderHeight{};
    uint32_t imageWidth_{};
    uint32_t imageHeight_{};
    // When the logical size last changed; the allocation shrinks only after
    // it has held for a while, so a drag never reallocates back and forth.
    std::chrono::steady_clock::time_point resized_{};
    FullOutputAovs fullOutputAovs_{};
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
    noorrhi::AccelerationStructure tlas;
    // The TLAS instance count the current tlas was built with.
    uint32_t tlasInstanceCount_{};
    // The scene clear this renderer's indices belong to.
    uint64_t publishedClearEpoch_{};
    // Textures below this index were uploaded (or failed to).
    std::size_t uploadedTextureCount_{};
    // The texture, by clear epoch and index, the environment's images were
    // made from; -1 for none.
    std::pair<uint64_t, int> environmentImageSource_{0, -1};

    // Mesh assets in the order instances first used them: an asset's index
    // here is the meshIndex its instances carry. Entries stay until the
    // scene is cleared.
    std::vector<const ::Mesh*> sourceMeshes_;
    std::unordered_map<const ::Mesh*, uint32_t> sourceMeshIndices_;
    // A mesh drawn with one scene material per slot. Instances drawing the
    // same mesh with the same materials share one: its hit records, its
    // material table and the BLAS for its materials' opacity. Bindings stay
    // until the scene is cleared, and their hit records follow in order.
    struct MaterialBinding
    {
        ::Mesh* mesh{};
        std::vector<uint32_t> materials;
        noorrhi::Buffer<uint32_t> materialTable;
        uint32_t hitRecordOffset{};
        std::vector<bool> opacity;
        noorrhi::AccelerationStructure blas;
        uint8_t mask{};
        // Some material's back faces are seen: camera rays cull no face.
        bool doubleSided{};
    };
    std::vector<MaterialBinding> bindings_;
    std::map<std::pair<const ::Mesh*, std::vector<uint32_t>>, uint32_t> bindingIndices_;
    // Binding indices by the scene material they draw with, and by mesh.
    std::vector<std::vector<uint32_t>> materialBindings_;
    std::unordered_map<const ::Mesh*, std::vector<uint32_t>> meshBindings_;
    std::vector<HitRecord> hitRecords_;
    bool hitRecordsChanged_{};
    // A binding's BLAS or hit records moved since the TLAS records were
    // written, so every record must be rewritten.
    bool bindingsMoved_{};
    // Material hit shaders by program, in the order renderers index them.
    std::unordered_map<const nr::materialx::MaterialShader*, uint32_t> materialShaderIndices_;
    std::vector<std::shared_ptr<const nr::materialx::MaterialShader>> materialShaderPrograms_;
    std::vector<MaterialHitShaders> materialShaders_;

    // Host copies and device tables: one pointer per material and per source
    // mesh, one instance per mesh instance slot and one TLAS record per
    // placement. Each table grows by doubling and receives only the entries
    // that changed.
    std::vector<std::uint64_t> materialPointerData_;
    std::vector<std::uint64_t> meshRecordData_;
    std::vector<nr::graphics::Instance> instanceData_;
    std::vector<noorrhi::InstanceRecord> tlasRecordData_;
    // The first TLAS record of each slot, and one past the last.
    std::vector<uint32_t> slotRecordOffsets_;
    // A slot's per-instance stream on the GPU: its color override or custom
    // data. The owner keeps the source alive, so an unchanged data pointer
    // means an unchanged stream.
    template <class T>
    struct SlotStream
    {
        std::shared_ptr<const void> owner;
        const T* data{};
        noorrhi::Buffer<T> buffer;
    };
    std::vector<SlotStream<uint32_t>> slotColors_;
    std::vector<SlotStream<float>> slotCustomData_;
    noorrhi::Buffer<std::uint64_t> meshRecords_;
    noorrhi::Buffer<nr::graphics::Instance> instances_;
    noorrhi::Buffer<noorrhi::InstanceRecord> tlasRecords_;

    nr::graphics::Scene sceneBuffers() const;
    void publishMaterials(Scene& scene, const std::vector<uint32_t>& changed);
    // Uploads a changed mesh and rebuilds the BLAS of its bindings.
    void publishMesh(const Scene& scene, ::Mesh& mesh);
    uint32_t sourceMesh(::Mesh& mesh);
    uint32_t binding(const Scene& scene, const MeshInstance& instance);
    // Brings the binding's BLAS and mask in line with its materials; returns
    // whether either changed.
    bool updateBindingOpacity(const Scene& scene, uint32_t binding);
    // Some section of the binding casts shadows.
    static bool bindingCastsShadow(const Scene& scene, const MaterialBinding& binding);
    static bool sectionCastsShadow(const Scene& scene, const MaterialBinding& binding,
        const MeshSection& section);
    void writeBindingHitRecords(const Scene& scene, uint32_t binding);
    // Lays out every binding's hit records again, after one's section count
    // changed.
    void layoutHitRecords(const Scene& scene);
    template <class T>
    std::uint64_t uploadSlotStream(SlotStream<T>& entry, std::span<const T> data,
        const std::shared_ptr<const void>& owner);
    std::uint64_t slotColors(uint32_t slot, const MeshInstance& instance);
    void publishInstances(const Scene& scene, std::vector<uint32_t> changedSlots, bool rewriteAll);
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
    // Sampled by any material whose texture failed to load at import time.
    noorrhi::Image<std::byte> whiteTexture;
};

// Renderer contract shared by the session and integrations. It owns no GPU
// resources; concrete renderers compose RaytracerResources as needed.
class Raytracer
{
public:
    virtual ~Raytracer() = default;
    virtual RaytracerType type() const noexcept = 0;
    virtual RaytracerResources& resources() = 0;
    virtual const RaytracerResources& resources() const = 0;
    virtual bool prepareFrameResources() = 0;
    virtual void render(uint32_t frameIndex = 0, uint32_t sampleIndex = 0) = 0;
    // Discards temporal state before the next frame. Stateless renderers do
    // not need to override this.
    virtual void restartTemporalHistory() {}
    virtual uint32_t traceWidth() const { return resources().width(); }
    virtual uint32_t traceHeight() const { return resources().height(); }
    virtual uint32_t readCryptomatteAtOutput(uint32_t x, uint32_t y)
    { return resources().readCryptomatteAt(x, y); }
    virtual noorrhi::float4 readPositionAtOutput(uint32_t x, uint32_t y)
    { return resources().readPositionAt(x, y); }

    void resize(uint32_t width, uint32_t height) { resources().resize(width, height); }
    nr::graphics::Frame& data() { return resources().data; }
    void commit() { resources().commit(); }
    void uploadScene(Scene& scene) { resources().uploadScene(scene); }
    void publishScene(Scene& scene, bool = true) { resources().publishScene(scene); }
    void applyRenderSettings(const RenderSettings& settings) { resources().applyRenderSettings(settings); }
    void updateLights(const Scene& scene, const Scene::LightIndices& changed)
    { resources().updateLights(scene, changed); }
    void updateCamera(const Scene& scene) { resources().updateCamera(scene); }
    void uploadEnvironment(Scene& scene) { resources().uploadEnvironment(scene); }
    double lastDispatchMilliseconds() { return resources().lastDispatchMilliseconds(); }
    noorrhi::ImageHandle outputImageHandle() const { return resources().outputImageHandle(); }
    noorrhi::TextureHandle outputTexture() const { return resources().outputTexture(); }
    noorrhi::TextureHandle outputSampledTexture() const { return resources().outputSampledTexture(); }
    noorrhi::TextureHandle albedoTexture() const { return resources().albedoTexture(); }
    noorrhi::TextureHandle normalTexture() const { return resources().normalTexture(); }
    noorrhi::TextureHandle positionTexture() const { return resources().positionTexture(); }
    noorrhi::TextureHandle cryptomatteTexture() const { return resources().cryptomatteTexture(); }
    noorrhi::GpuPtr<std::uint32_t> gaussianOverdrawPtr() const { return resources().gaussianOverdrawPtr(); }
    noorrhi::Device& device() const { return resources().device(); }
    std::vector<std::byte> readColor() { return resources().readColor(); }
    std::vector<noorrhi::float4> readBeauty() { return resources().readBeauty(); }
    std::vector<std::uint32_t> readCryptomatte() { return resources().readCryptomatte(); }
    std::vector<noorrhi::float4> readPosition() { return resources().readPosition(); }
    uint32_t width() const { return resources().width(); }
    uint32_t height() const { return resources().height(); }
    uint32_t imageWidth() const { return resources().imageWidth(); }
    uint32_t imageHeight() const { return resources().imageHeight(); }
    virtual bool linkCompiledMaterialShaders() { return false; }
    virtual MaterialShaderStage materialShaderStage() const { return MaterialShaderStage::Idle; }
    virtual void waitForMaterialShaders() {}
};
