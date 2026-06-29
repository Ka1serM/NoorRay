#pragma once

#include "../Vulkan/Context.h"
#include "../Vulkan/Image.h"
#include <array>
#include <cstdint>

class Tonemapper {
public:
    Tonemapper(Context& context, uint32_t width, uint32_t height,
               const Image& inputImage0, const Image& inputImage1,
               const Image& idImage0, const Image& idImage1,
               vk::Format outputImageFormat);
    ~Tonemapper();

    void dispatch(vk::CommandBuffer commandBuffer, uint32_t bufferIndex, uint32_t selectedIndex);
    void resize(uint32_t width, uint32_t height,
                const Image& inputImage0, const Image& inputImage1,
                const Image& idImage0, const Image& idImage1,
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

    void writeDescriptors(uint32_t bufferIndex, const Image& inputImage, const Image& idImage);
};
