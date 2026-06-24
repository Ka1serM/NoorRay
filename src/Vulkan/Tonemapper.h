#pragma once

#include "../Vulkan/Context.h"
#include "../Vulkan/Image.h"

class Tonemapper {
public:
    Tonemapper(Context& context, uint32_t width, uint32_t height, const Image& inputImage, vk::Format outputImageFormat);
    ~Tonemapper();

    void dispatch(vk::CommandBuffer commandBuffer);
    void resize(uint32_t width, uint32_t height, const Image& inputImage, vk::Format outputImageFormat);
    Image& getOutputImage() { return outputImage; }

private:
    Context& context;
    Image outputImage;
    vk::UniqueShaderModule shaderModule;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline pipeline;
    vk::UniqueDescriptorSet descriptorSet;

    void writeDescriptors(const Image& inputImage);
};
