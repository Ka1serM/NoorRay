#pragma once

#include "Shaders/Shared.h"
#include "Vulkan/Buffer.h"
#include "Vulkan/Context.h"

class RayLutGenerator
{
public:
    RayLutGenerator(Context& context);
    ~RayLutGenerator();

    void writeDescriptors(const Buffer& sceneSettingsBuffer, const Buffer& rayLutBuffer);
    void dispatch(const vk::CommandBuffer& commandBuffer, uint32_t width, uint32_t height);

private:
    struct RayLutPushData
    {
        int width;
        int height;
        int _pad0;
        int _pad1;
    };

    Context& context;
    vk::UniqueShaderModule shaderModule;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline pipeline;
    vk::UniqueDescriptorSet descriptorSet;
};
