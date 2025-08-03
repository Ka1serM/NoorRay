#pragma once

#include "../Vulkan/Context.h"
#include "../Vulkan/Image.h"

class Tonemapper {
public:
    Tonemapper(Context& context, uint32_t width, uint32_t height, const Image& inputImage);
    ~Tonemapper();

    void dispatch(vk::CommandBuffer commandBuffer);
    // Return a non-const reference to allow layout transitions on the image
    Image& getOutputImage() { return outputImage; }

private:
    Image outputImage;
    vk::UniqueShaderModule shaderModule;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline pipeline;
    vk::UniqueDescriptorSet descriptorSet;
};
