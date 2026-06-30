#pragma once

#include "../Vulkan/Context.h"
#include "../Vulkan/Image.h"
#include <array>
#include <cstdint>

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
        float exposure,
        int bufferVisualization,
        int tonemappingEnabled);
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
    vk::UniqueShaderModule shaderModule;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline pipeline;
    std::array<vk::UniqueDescriptorSet, 2> descriptorSets;

    void writeDescriptors(uint32_t bufferIndex,
        const Image& color, const Image& albedo, const Image& normal,
        const Image& crypto, const Image& position);
};
