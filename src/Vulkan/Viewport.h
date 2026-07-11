#pragma once

#include "../Vulkan/Context.h"
#include "../Vulkan/Image.h"
#include "../Vulkan/Buffer.h"
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
constexpr float ViewportBillboardPixelRadius = 24.0f;

class Viewport {
public:
    Viewport(Context& context, uint32_t width, uint32_t height,
               const Image& color0,    const Image& color1,
               const Image& albedo0,   const Image& albedo1,
               const Image& normal0,   const Image& normal1,
               const Image& crypto0,   const Image& crypto1,
               const Image& position0, const Image& position1,
               vk::Format outputImageFormat);
    ~Viewport();

    void dispatch(
        vk::CommandBuffer commandBuffer,
        uint32_t bufferIndex,
        uint32_t selectedIndex,
        const glm::mat4& viewProjection,
        float exposure,
        int bufferVisualization,
        int tonemappingEnabled,
        bool showBillboards = true);
    // Rebuilds the billboard list from the scene's current objects (cheap: a
    // handful of structs memcpy'd into a host-visible buffer). Call once per
    // dispatched frame before dispatch().
    void updateBillboards(const Scene& scene);
    void resize(uint32_t width, uint32_t height,
                const Image& color0,    const Image& color1,
                const Image& albedo0,   const Image& albedo1,
                const Image& normal0,   const Image& normal1,
                const Image& crypto0,   const Image& crypto1,
                const Image& position0, const Image& position1,
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
    std::array<vk::UniqueDescriptorSet, 2> descriptorSets;

    // Billboard overlay — a tiny raster pass (dynamic rendering, instanced quads)
    // drawn on top of the compute pass's output.
    vk::UniqueShaderModule billboardShaderModule;
    vk::UniqueDescriptorSetLayout billboardDescriptorSetLayout;
    vk::UniquePipelineLayout billboardPipelineLayout;
    vk::UniquePipeline billboardPipeline;
    vk::UniqueDescriptorSet billboardDescriptorSet;
    Buffer billboardBuffer;
    uint32_t billboardCapacity{};
    uint32_t billboardCount{};

    void writeDescriptors(uint32_t bufferIndex,
        const Image& color, const Image& albedo, const Image& normal,
        const Image& crypto, const Image& position);
    void createBillboardPipeline();
    void reserveBillboards(uint32_t capacity);
    void writeBillboardDescriptor();
    void drawBillboards(vk::CommandBuffer commandBuffer, const glm::mat4& viewProjection);
};
