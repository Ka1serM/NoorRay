#pragma once

#include <noorrhi/noorrhi.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Scene/Handle.h"
#include "Scene/Scene.h"
#include "Scene/VolumeInstance.h"
#include "Shared/Viewport.h"

class Scene;

// Fixed-size screen-space icon drawn by the viewport shader for a scene object
// that carries a Billboard. Kept separate from the physics light structs
// (PointLight etc.) so its GPU layout is simple and stable regardless of what
// kind of object it represents.
using ViewportBillboard = nr::graphics::ViewportBillboard;

// Fixed 80 cm square; the icon fills about three quarters of its texture.
constexpr float ViewportBillboardHalfSize = 40.0f;

// The AOV images the composite pass reads. These are descriptor-heap handles of
// images the raytracer created through noorrhi::Device; nothing here is a
// descriptor or an index.
struct ViewportInputs
{
    noorrhi::TextureHandle color{};
    // Only their own buffer views read these, which show black while the
    // renderer does not write them (a zero handle).
    noorrhi::TextureHandle albedo{};
    noorrhi::TextureHandle normal{};
    // At the render resolution, sampled with `depthJitter` like `depth`.
    noorrhi::TextureHandle crypto{};
    noorrhi::TextureHandle position{};
    noorrhi::TextureHandle depth{};
    // The render samples' offset in render pixels, for `depth` and `crypto`.
    glm::vec2 depthJitter{};

    // What every view needs.
    explicit operator bool() const noexcept
    {
        return color && crypto;
    }
};

// A host-owned presentation image. Interactive hosts can render the final
// viewport composite directly into a frame slot they will later sample.
struct ViewportOutput
{
    noorrhi::ImageHandle image{};
    noorrhi::TextureHandle storage{};
    uint32_t width{};
    uint32_t height{};
    noorrhi::ImageFormat format{noorrhi::ImageFormat::Auto};

    explicit operator bool() const noexcept { return image && storage && width && height; }
};

class Viewport {
public:
    // width/height is the logical render size; traceWidth/traceHeight is the
    // resolution the raytracer actually traced (equal to the logical size
    // unless it upscales); imageWidth/imageHeight is the allocation of the
    // raytracer's AOV images, which the output matches.
    Viewport(noorrhi::Device& gpu_device, uint32_t width, uint32_t height,
             uint32_t traceWidth, uint32_t traceHeight,
             uint32_t imageWidth, uint32_t imageHeight,
             const ViewportInputs& inputs, noorrhi::ImageFormat outputImageFormat,
             bool exportOutputMemory = false, bool externalOutput = false);
    ~Viewport();

    // Recorded into whatever noorrhi::Frame is open around the call.
    // The selection outline averages its estimate over frames independently
    // of beauty accumulation; `restartOutline` clears that history after the
    // camera or scene changes.
    void dispatch(
        uint32_t selectedCryptomatteId,
        bool restartOutline,
        const glm::mat4& viewProjection,
        float exposure,
        int bufferVisualization,
        bool tonemappingEnabled,
        bool showBillboards = true,
        SceneObjectHandle selectedObject = {}, ViewportOutput output = {});
    // Refreshes the overlay records the scene changed: billboard icons and
    // volume outlines. Calling this each frame is cheap when nothing changed;
    // a moved, edited or hidden icon rewrites only its own record. Call
    // outside an open frame.
    void updateOverlays(Scene& scene);
    // Icons are deliberately an always-on-top editing overlay. Keep their
    // useful range local to the active camera instead of depth-testing them
    // against scene geometry.
    void setBillboardCameraPosition(glm::vec3 position) { billboardCameraPosition_ = position; hasBillboardCamera_ = true; }
    void clearBillboardCameraPosition() { hasBillboardCamera_ = false; }
    void setDepthJitter(glm::vec2 jitter) { inputs.depthJitter = jitter; }
    // The object whose icon covers output pixel (x, y) (bottom-left origin),
    // read back from the billboard-id buffer the last dispatch stamped. Invalid /
    // nothing where no icon was drawn, including while icons are hidden.
    SceneObjectHandle billboardObjectAt(uint32_t x, uint32_t y) const;
    SceneObjectHandle volumeObjectAt(uint32_t x, uint32_t y) const;
    std::optional<glm::vec3> billboardPositionAt(uint32_t x, uint32_t y) const;
    // Replaces the output image only when the allocation or format changes;
    // a new logical size alone is free.
    void resize(uint32_t width, uint32_t height,
                uint32_t traceWidth, uint32_t traceHeight,
                uint32_t imageWidth, uint32_t imageHeight,
                const ViewportInputs& inputs, noorrhi::ImageFormat outputImageFormat);

    // The composited viewport is the public render result. It includes the
    // selected AOV visualization, tonemapping, and optional scene billboards.
    // Consumers can sample it through NoorRHI or obtain its native image identity
    // for interop with a UI renderer such as Dear ImGui.
    noorrhi::ImageHandle outputImageHandle() const { return outputImage.handle(); }
    noorrhi::TextureHandle outputTexture() const { return outputImage.sampled_handle(); }
    noorrhi::TextureHandle outputStorageTexture() const { return outputImage.storage_handle(); }
    noorrhi::ImageFormat outputFormat() const { return outputFormat_; }
    // Logical size of the composited render.
    uint32_t outputWidth() const { return logicalWidth; }
    uint32_t outputHeight() const { return logicalHeight; }
    // Allocated size of the output image. The render occupies its bottom-left
    // outputWidth() x outputHeight() texels.
    uint32_t outputImageWidth() const { return outputImageWidth_; }
    uint32_t outputImageHeight() const { return outputImageHeight_; }
    std::vector<noorrhi::float4> readOutput() const;

private:
    noorrhi::Device& gpuDevice;
    noorrhi::Image<std::byte> outputImage;
    bool externalOutput_{};
    // Per-pixel selection distance field: the running average of every
    // frame's estimate since the outline's accumulation last restarted.
    noorrhi::Image<std::byte> selectionSdfImage;
    uint32_t outlinedCryptomatteId_ = ~0u;
    uint32_t outlineSampleCount_ = 0;
    uint32_t logicalWidth{};
    uint32_t logicalHeight{};
    uint32_t outputImageWidth_{};
    uint32_t outputImageHeight_{};
    // The resolution behind the AOVs. Only their silhouette sharpness depends
    // on it, so it is a plain value with no resource attached.
    uint32_t traceWidth_{};
    uint32_t traceHeight_{};
    noorrhi::ImageFormat outputFormat_ = noorrhi::ImageFormat::Rgba32Float;
    bool exportOutputMemory_{};
    ViewportInputs inputs{};

    // Beauty/AOV composite - compute pass.
    noorrhi::Shader shader;
    noorrhi::ComputePipeline pipeline;

    // Billboard overlay - a tiny raster pass (dynamic rendering, instanced quads)
    // drawn on top of the compute pass's output.
    noorrhi::Shader billboardVertexShader;
    noorrhi::Shader billboardFragmentShader;
    noorrhi::GraphicsPipeline billboardPipeline;
    // One distance-field texture per icon, and the heap indices the shader reads.
    noorrhi::Sampler billboardSampler;
    std::vector<noorrhi::Image<std::byte>> billboardTextures;
    noorrhi::Buffer<std::uint32_t> billboardTextureHandles;
    // Billboard picking: one uint per output pixel, billboard index + 1 or 0,
    // cleared and stamped by two compute passes after the icons are drawn.
    noorrhi::Buffer<std::uint32_t> lightIdBuffer;
    noorrhi::Shader lightIdClearShader;
    noorrhi::Shader lightIdStampShader;
    noorrhi::ComputePipeline lightIdClearPipeline;
    noorrhi::ComputePipeline lightIdStampPipeline;
    noorrhi::Buffer<std::byte> billboardBuffer;
    std::vector<ViewportBillboard> billboardData;
    // The object each billboard record was built from, by record index.
    std::vector<SceneObjectHandle> billboardHandles;
    uint32_t billboardCapacity{};
    noorrhi::GpuPtr<std::byte> billboardEntry{};
    uint32_t billboardCount{};
    glm::vec3 billboardCameraPosition_{};
    bool hasBillboardCamera_{};

    // Volume outlines: one instanced draw per volume, sharing edge buffers.
    struct VolumeDraw {
        SceneObjectHandle handle;
        std::shared_ptr<const VolumeOutline> outline;
        glm::mat4 world;
        glm::vec4 color;
        noorrhi::GpuPtr<glm::vec3> segments;
    };
    noorrhi::Shader volumeVertexShader;
    noorrhi::Shader volumeFragmentShader;
    noorrhi::GraphicsPipeline volumePipeline;
    std::vector<VolumeDraw> volumeDraws;
    std::vector<uint32_t> volumeSlotToDraw;
    noorrhi::Buffer<std::uint32_t> volumeIdBuffer;
    // Whether the id buffers hold nothing, so a frame that draws into neither
    // need not clear them.
    bool lightIdsClear_{};
    bool volumeIdsClear_{};
    uint64_t observedVolumeStructureRevision{};

    void createOutputResources(uint32_t width, uint32_t height, noorrhi::ImageFormat format);
    void createBillboardPipeline();
    void createBillboardTextures();
    void createVolumePipeline();
    // Returns whether the buffer was replaced, which drops its contents.
    bool reserveBillboards(uint32_t capacity);
    void updateBillboards(Scene& scene);
    void updateVolumes(Scene& scene);
    std::optional<uint32_t> billboardAt(uint32_t x, uint32_t y) const;
    void drawBillboards(const glm::mat4& viewProjection, SceneObjectHandle selectedObject,
        const ViewportOutput& output);
    void drawVolumes(const glm::mat4& viewProjection, SceneObjectHandle selectedObject,
        const ViewportOutput& output);
};
