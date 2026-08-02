#pragma once

#include "Backend/Vulkan/Runtime/Context.h"
#include "Backend/Vulkan/Runtime/Image.h"
#include "Backend/CUDA/Unique/SharedVector.h"
#include "Backend/CUDA/Unique/SharedBuffer.h"
#include <array>
#include <cstdint>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
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

class Viewport {
public:
    Viewport(Context& context, uint32_t width, uint32_t height,
             const Image& color,    const Image& albedo,
             const Image& normal,   const Image& crypto,
             const Image& position,
             const nr::cuda::UniqueSharedBuffer& denoised,
             vk::Format outputImageFormat);
    ~Viewport();

    void dispatch(
        vk::CommandBuffer commandBuffer,
        uint32_t selectedCryptomatteId,
        const glm::mat4& viewProjection,
        float exposure,
        int bufferVisualization,
        bool tonemappingEnabled,
        bool showBillboards = true);
    // Refreshes the persistent overlay buffer only after a light mutation.
    // Calling this each frame is an O(1) revision check in the common case.
    void updateBillboards(const Scene& scene);
    void resize(uint32_t width, uint32_t height,
                const Image& color,    const Image& albedo,
                const Image& normal,   const Image& crypto,
                const Image& position,
                const nr::cuda::UniqueSharedBuffer& denoised,
                vk::Format outputImageFormat);
    Image& getOutputImage() { return outputImage; }

private:
    Context& context;
    Image outputImage;

    // Beauty/AOV composite — compute pass.
    vk::UniqueShaderModule shaderModule;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline pipeline;
    std::array<vk::UniqueDescriptorSet, 1> descriptorSets;
    // False until every storage image the compute pass binds exists.
    bool descriptorsValid{};

    // Billboard overlay — a tiny raster pass (dynamic rendering, instanced quads)
    // drawn on top of the compute pass's output.
    vk::UniqueShaderModule billboardShaderModule;
    vk::UniqueDescriptorSetLayout billboardDescriptorSetLayout;
    vk::UniquePipelineLayout billboardPipelineLayout;
    vk::UniquePipeline billboardPipeline;
    vk::UniqueDescriptorSet billboardDescriptorSet;
    nr::cuda::SharedVector<ViewportBillboard> billboards;
    uint32_t billboardCount{};
    uint64_t observedLightRevision{};

    void writeDescriptors(
        const Image& color, const Image& albedo, const Image& normal,
        const Image& crypto, const Image& position,
        const nr::cuda::UniqueSharedBuffer& denoised);
    void createBillboardPipeline();
    void reserveBillboards(uint32_t capacity);
    void writeBillboardDescriptor();
    void drawBillboards(vk::CommandBuffer commandBuffer, const glm::mat4& viewProjection);
};
