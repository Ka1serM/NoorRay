#pragma once

#include "Context.h"
#include "Image.h"

class HdrToLdrCompute {
public:
    HdrToLdrCompute(Context& context, uint32_t width, uint32_t height, vk::ImageView inputImageView);
    void dispatch(vk::CommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z);
    Image outputImage;
private:
    vk::UniqueShaderModule shaderModule;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline pipeline;
    vk::UniqueDescriptorPool descriptorPool;
    vk::UniqueDescriptorSet descriptorSet;
};
