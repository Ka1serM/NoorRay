#pragma once

#include "Context.h"
#include "Image.h"

class Tonemapper {
public:
    Tonemapper(Context& context, uint32_t width, uint32_t height, vk::ImageView inputImageView);
    ~Tonemapper();
    Tonemapper(const Tonemapper&) = delete;
    Tonemapper& operator=(const Tonemapper&) = delete;
    Tonemapper(Tonemapper&&) = delete;
    Tonemapper& operator=(Tonemapper&&) = delete;
    
    void dispatch(vk::CommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z);
    Image outputImage;
private:
    vk::UniqueDescriptorSet descriptorSet;
    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueShaderModule shaderModule;
};