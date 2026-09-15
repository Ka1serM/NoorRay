#pragma once

#include <gpu/gpu.hpp>

#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Shared/Viewport.h"

class Scene;

// Fixed-size screen-space gizmo drawn by the viewport shader for a scene object
// (currently lights). Kept separate from the physics light structs (PointLight etc.)
// so its GPU layout is simple and stable regardless of what kind of object it
// represents.
using ViewportBillboard = nr::graphics::ViewportBillboard;

// Fixed screen-space half-size of a billboard icon, in pixels. Shared with
// ViewportPanel's click-picking radius so hit-testing matches what's drawn.
constexpr float ViewportBillboardPixelRadius = 32.0f;

// Projects a billboard's world position into the renderer's bottom-left pixel
// space - the same space ViewportPanel::screenToPixel reports clicks in, and
// the one the overlay raster pass draws into (Y-up NDC into an unflipped
// graphics API viewport, presented with a single V flip). Hit-testing a click against
// the drawn icon only agrees when both sides use this mapping. Returns false
// when the billboard sits behind the camera and is not drawn at all.
inline bool projectViewportBillboard(const glm::mat4& viewProjection,
    const glm::vec3& worldPosition, const uint32_t width, const uint32_t height,
    glm::vec2& pixel)
{
    const glm::vec4 clip = viewProjection * glm::vec4(worldPosition, 1.0f);
    if (clip.w <= 0.0f)
        return false;
    const glm::vec2 ndc = glm::vec2(clip) / clip.w;
    pixel = glm::vec2((ndc.x * 0.5f + 0.5f) * static_cast<float>(width),
        (ndc.y * 0.5f + 0.5f) * static_cast<float>(height));
    return true;
}

// The AOV images the composite pass reads. These are descriptor-heap handles of
// images the raytracer created through gpu::Device, plus the device address of
// the overdraw counter; nothing here is a descriptor or an index.
struct ViewportInputs
{
    gpu::TextureHandle color{};
    gpu::TextureHandle albedo{};
    gpu::TextureHandle normal{};
    gpu::TextureHandle crypto{};
    gpu::TextureHandle position{};
    gpu::GpuPtr<std::uint32_t> overdraw{};

    explicit operator bool() const noexcept
    {
        return color && albedo && normal && crypto && position && overdraw.address;
    }
};

class Viewport {
public:
    Viewport(gpu::Device& gpu_device, uint32_t width, uint32_t height,
             const ViewportInputs& inputs, gpu::ImageFormat outputImageFormat,
             bool exportOutputMemory = false);
    ~Viewport();

    // Recorded into whatever gpu::Frame is open around the call.
    void dispatch(
        uint32_t selectedCryptomatteId,
        const glm::mat4& viewProjection,
        float exposure,
        int bufferVisualization,
        int gaussianOverdrawMax,
        bool tonemappingEnabled,
        bool showBillboards = true);
    // Refreshes the persistent overlay buffer only after a light mutation.
    // Calling this each frame is an O(1) revision check in the common case.
    void updateBillboards(const Scene& scene);
    void resize(uint32_t width, uint32_t height, const ViewportInputs& inputs,
                gpu::ImageFormat outputImageFormat);

    // The composited viewport is the public render result. It includes the
    // selected AOV visualization, tonemapping, and optional scene billboards.
    // Consumers can sample it through gpu or obtain its native image identity
    // for interop with a UI renderer such as Dear ImGui.
    gpu::ImageHandle outputImageHandle() const { return outputImage.handle(); }
    gpu::TextureHandle outputTexture() const { return outputImage.sampled_handle(); }
    gpu::TextureHandle outputStorageTexture() const { return outputImage.storage_handle(); }
    gpu::ImageFormat outputFormat() const { return outputFormat_; }
    uint32_t outputWidth() const { return outputImage.width(); }
    uint32_t outputHeight() const { return outputImage.height(); }
    std::vector<gpu::float4> readOutput() const;

private:
    gpu::Device& gpuDevice;
    gpu::Image<std::byte> outputImage;
    gpu::ImageFormat outputFormat_ = gpu::ImageFormat::Rgba32Float;
    bool exportOutputMemory_{};
    ViewportInputs inputs{};

    // Beauty/AOV composite - compute pass.
    gpu::Shader shader;
    gpu::ComputePipeline pipeline;

    // Billboard overlay - a tiny raster pass (dynamic rendering, instanced quads)
    // drawn on top of the compute pass's output.
    gpu::Shader billboardVertexShader;
    gpu::Shader billboardFragmentShader;
    gpu::GraphicsPipeline billboardPipeline;
    gpu::Buffer<std::byte> billboardBuffer;
    std::vector<ViewportBillboard> billboardData;
    uint32_t billboardCapacity{};
    gpu::GpuPtr<std::byte> billboardEntry{};
    uint32_t billboardCount{};
    uint64_t observedLightRevision{};

    void createOutputImage(uint32_t width, uint32_t height, gpu::ImageFormat format);
    void createBillboardPipeline();
    void reserveBillboards(uint32_t capacity);
    void drawBillboards(const glm::mat4& viewProjection);
};
