#pragma once

#include "Context.h"
#include "Image.h"
#include <vulkan/vulkan.hpp>

class Tonemapper {
private:
    Image outputImage;
    vk::UniqueDescriptorSet descriptorSet;
    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueShaderModule shaderModule;

public:
    Tonemapper(Context& context, uint32_t width, uint32_t height, const Image& inputImage);
    ~Tonemapper();
    
    void dispatch(vk::CommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z);
    const Image& getOutputImage() const { return outputImage; }
};