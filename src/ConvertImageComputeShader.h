#pragma once

#include "Context.h"

class ConvertImageComputeShader {
public:
    ConvertImageComputeShader(const std::string &shaderPath, Context &context, vk::ImageView inputImageView, vk::ImageView outputImageView);

    void dispatch(vk::CommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z);

private:
    Context& context;

    vk::UniqueShaderModule shaderModule;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline pipeline;
    vk::DescriptorPool descriptorPool;
    vk::DescriptorSet descriptorSet;
};
