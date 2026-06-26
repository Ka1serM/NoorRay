#include "RayLutGenerator.h"

#include "Log.h"

#include <array>

RayLutGenerator::RayLutGenerator(Context& context)
    : context(context)
{
    static constexpr unsigned char code[] = {
        #embed "../Shaders/Wavefront/GenerateRayLut.spv"
    };
    shaderModule = context.getDevice().createShaderModuleUnique({{}, sizeof(code), reinterpret_cast<const uint32_t*>(code)});

    std::array<vk::DescriptorSetLayoutBinding, 2> bindings {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    vk::DescriptorSetLayoutCreateInfo descriptorLayoutInfo{};
    descriptorLayoutInfo.setBindings(bindings);
    descriptorSetLayout = context.getDevice().createDescriptorSetLayoutUnique(descriptorLayoutInfo);

    vk::PushConstantRange pcRange{};
    pcRange.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    pcRange.setSize(sizeof(RayLutPushData));

    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.setSetLayouts(descriptorSetLayout.get());
    layoutInfo.setPushConstantRanges(pcRange);
    pipelineLayout = context.getDevice().createPipelineLayoutUnique(layoutInfo);

    vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, *shaderModule, "main");
    pipeline = context.getDevice().createComputePipelineUnique({}, {{}, stageInfo, *pipelineLayout}).value;

    vk::DescriptorSetLayout layout = descriptorSetLayout.get();
    vk::DescriptorSetAllocateInfo allocInfo(context.getDescriptorPool(), 1, &layout);
    descriptorSet = std::move(context.getDevice().allocateDescriptorSetsUnique(allocInfo).front());
}

RayLutGenerator::~RayLutGenerator()
{
    LOG_INFO("Destroying RayLutGenerator");
}

void RayLutGenerator::writeDescriptors(const Buffer& sceneSettingsBuffer, const Buffer& rayLutBuffer)
{
    std::array<vk::DescriptorBufferInfo, 2> infos {
        sceneSettingsBuffer.getDescriptorInfo(),
        rayLutBuffer.getDescriptorInfo(),
    };

    std::array<vk::WriteDescriptorSet, 2> writes {
        vk::WriteDescriptorSet{}
            .setDstSet(descriptorSet.get())
            .setDstBinding(0)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setBufferInfo(infos[0]),
        vk::WriteDescriptorSet{}
            .setDstSet(descriptorSet.get())
            .setDstBinding(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setBufferInfo(infos[1]),
    };
    context.getDevice().updateDescriptorSets(writes, {});
}

void RayLutGenerator::dispatch(const vk::CommandBuffer& commandBuffer, uint32_t width, uint32_t height)
{
    RayLutPushData pc{};
    pc.width = static_cast<int>(width);
    pc.height = static_cast<int>(height);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.get());
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, descriptorSet.get(), {});
    commandBuffer.pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(RayLutPushData), &pc);
    const uint32_t groupCountX = (width + GROUP_SIZE - 1u) / GROUP_SIZE;
    const uint32_t groupCountY = (height + GROUP_SIZE - 1u) / GROUP_SIZE;
    commandBuffer.dispatch(groupCountX, groupCountY, 1);
}
