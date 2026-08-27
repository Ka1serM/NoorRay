#pragma once

#include <gpu/gpu.hpp>

#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

class Scene;

// Fixed-size screen-space gizmo drawn by the viewport shader for a scene object
// (currently lights). Kept separate from the physics light structs (PointLight etc.)
// so its GPU layout is simple and stable regardless of what kind of object it
// represents.
struct ViewportBillboard
{
    glm::vec4 positionType{}; // xyz = world position, w = object type (LightInstance::Type*)
    glm::vec4 color{};        // rgb = display color
};

// Fixed screen-space half-size of a billboard icon, in pixels. Shared with
// ViewportPanel's click-picking radius so hit-testing matches what's drawn.
constexpr float ViewportBillboardPixelRadius = 32.0f;

// Projects a billboard's world position into the renderer's bottom-left pixel
// space - the same space ViewportPanel::screenToPixel reports clicks in, and
// the one the overlay raster pass draws into (Y-up NDC into an unflipped
// Vulkan viewport, presented with a single V flip). Hit-testing a click against
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

// The AOV images the composite pass reads. They are storage handles of images
// the raytracer created through gpu::Device, so no descriptors are written
// here - the handles already are the heap indices the shader indexes with.
struct ViewportInputs
{
    gpu::ImageHandle color{};
    gpu::ImageHandle albedo{};
    gpu::ImageHandle normal{};
    gpu::ImageHandle crypto{};
    gpu::ImageHandle position{};
    gpu::ResourceHandle overdraw{};

    explicit operator bool() const noexcept
    {
        return color && albedo && normal && crypto && position && overdraw;
    }
};

class Viewport {
public:
    Viewport(gpu::Device& gpu_device, uint32_t width, uint32_t height,
             const ViewportInputs& inputs, gpu::ImageFormat outputImageFormat);
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
    const gpu::Image<std::byte>& getOutputImage() const { return outputImage; }

private:
    gpu::Device& gpuDevice;
    gpu::Image<std::byte> outputImage;
    gpu::ImageFormat outputFormat = gpu::ImageFormat::Rgba8Unorm;
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
    gpu::ResourceHandle billboardEntry{};
    uint32_t billboardCount{};
    uint64_t observedLightRevision{};

    void createOutputImage(uint32_t width, uint32_t height, gpu::ImageFormat format);
    void createBillboardPipeline();
    void reserveBillboards(uint32_t capacity);
    void drawBillboards(const glm::mat4& viewProjection);
};
