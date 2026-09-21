#pragma once

#include <noorrhi/noorrhi.hpp>

#include <cstdint>
#include <optional>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Scene/Handle.h"
#include "Shared/Viewport.h"

class Scene;

// Fixed-size screen-space gizmo drawn by the viewport shader for a scene object
// (currently lights). Kept separate from the physics light structs (PointLight etc.)
// so its GPU layout is simple and stable regardless of what kind of object it
// represents.
using ViewportBillboard = nr::graphics::ViewportBillboard;

// Screen-space half-size of a billboard icon, in pixels. Icons at or closer
// than the near distance (view-space, scene units) draw at the maximum size and
// shrink linearly to the minimum at the far distance.
constexpr float ViewportBillboardMinPixelRadius = 18.0f;
constexpr float ViewportBillboardMaxPixelRadius = 32.0f;
constexpr float ViewportBillboardNearDistance = 2.0f;
constexpr float ViewportBillboardFarDistance = 60.0f;
// Fraction of the icon's half-size stamped into the light-id buffer for
// picking. The glyphs reach about 0.7 of the quad, so this covers them without
// the corners.
constexpr float ViewportBillboardPickScale = 0.8f;

// The AOV images the composite pass reads. These are descriptor-heap handles of
// images the raytracer created through noorrhi::Device, plus the device address of
// the overdraw counter; nothing here is a descriptor or an index.
struct ViewportInputs
{
    noorrhi::TextureHandle color{};
    noorrhi::TextureHandle albedo{};
    noorrhi::TextureHandle normal{};
    noorrhi::TextureHandle crypto{};
    noorrhi::TextureHandle position{};
    noorrhi::GpuPtr<std::uint32_t> overdraw{};

    explicit operator bool() const noexcept
    {
        return color && albedo && normal && crypto && position && overdraw.address;
    }
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
             bool exportOutputMemory = false);
    ~Viewport();

    // Recorded into whatever noorrhi::Frame is open around the call.
    // The selection outline averages its estimate over frames, like the
    // renderer accumulates samples; `restartOutline` says the renderer's
    // accumulation restarted (the camera or the scene changed), so the
    // outline's does too.
    void dispatch(
        uint32_t selectedCryptomatteId,
        bool restartOutline,
        const glm::mat4& viewProjection,
        float exposure,
        int bufferVisualization,
        int gaussianOverdrawMax,
        bool tonemappingEnabled,
        bool showBillboards = true,
        SceneObjectHandle selectedObject = {});
    // Refreshes the persistent overlay buffer only after a light mutation.
    // Calling this each frame is an O(1) revision check in the common case; a
    // moved or edited light rewrites only the records that changed, and only
    // adding or removing scene objects rebuilds the list.
    void updateBillboards(const Scene& scene);
    // The light whose icon covers output pixel (x, y) (bottom-left origin),
    // read back from the light-id buffer the last dispatch stamped. Invalid /
    // nothing where no icon was drawn, including while icons are hidden.
    SceneObjectHandle lightAt(uint32_t x, uint32_t y) const;
    std::optional<glm::vec3> lightPositionAt(uint32_t x, uint32_t y) const;
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
    uint32_t outputImageWidth() const { return outputImage.width(); }
    uint32_t outputImageHeight() const { return outputImage.height(); }
    std::vector<noorrhi::float4> readOutput() const;

private:
    noorrhi::Device& gpuDevice;
    noorrhi::Image<std::byte> outputImage;
    // Per-pixel selection distance field: the running average of every
    // frame's estimate since the outline's accumulation last restarted.
    noorrhi::Image<std::byte> selectionSdfImage;
    uint32_t outlinedCryptomatteId_ = ~0u;
    uint32_t outlineSampleCount_ = 0;
    uint32_t logicalWidth{};
    uint32_t logicalHeight{};
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
    // Light picking: one uint per output pixel, billboard index + 1 or 0,
    // cleared and stamped by two compute passes after the icons are drawn.
    noorrhi::Buffer<std::uint32_t> lightIdBuffer;
    noorrhi::Shader lightIdClearShader;
    noorrhi::Shader lightIdStampShader;
    noorrhi::ComputePipeline lightIdClearPipeline;
    noorrhi::ComputePipeline lightIdStampPipeline;
    noorrhi::Buffer<std::byte> billboardBuffer;
    std::vector<ViewportBillboard> billboardData;
    // The light each billboard record was built from, by record index.
    std::vector<SceneObjectHandle> billboardHandles;
    uint32_t billboardCapacity{};
    noorrhi::GpuPtr<std::byte> billboardEntry{};
    uint32_t billboardCount{};
    uint64_t observedLightRevision{};
    uint64_t observedHierarchyRevision{};

    void createOutputImage(uint32_t width, uint32_t height, noorrhi::ImageFormat format);
    void createBillboardPipeline();
    void reserveBillboards(uint32_t capacity);
    void rebuildBillboards(const Scene& scene);
    std::optional<uint32_t> billboardAt(uint32_t x, uint32_t y) const;
    void drawBillboards(const glm::mat4& viewProjection, SceneObjectHandle selectedObject);
};
