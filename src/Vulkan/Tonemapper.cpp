#include "Tonemapper.h"
#include <iostream>

#include "Globals.h"
#include "Log.h"
#include "Shaders/Shared.h"

Tonemapper::Tonemapper(Context& context, const uint32_t width, const uint32_t height, const Image& inputImage, const vk::Format outputImageFormat)
: context(context),
  outputImage(context, width, height, outputImageFormat,  vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc |vk::ImageUsageFlagBits::eTransferDst)
{
    //Load shader
    static constexpr unsigned char code[] = {
        #embed "../Shaders/Tonemapping/Tonemapper.spv"
    };
    shaderModule = context.getDevice().createShaderModuleUnique({{}, sizeof(code), reinterpret_cast<const uint32_t*>(code)});

    // Descriptor set layout with 2 storage images
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}
    };

    descriptorSetLayout = context.getDevice().createDescriptorSetLayoutUnique({{}, static_cast<uint32_t>(bindings.size()), bindings.data()});

    pipelineLayout = context.getDevice().createPipelineLayoutUnique({{}, 1, &*descriptorSetLayout});

    // Compute pipeline
    vk::PipelineShaderStageCreateInfo shaderStage({}, vk::ShaderStageFlagBits::eCompute, *shaderModule, "main");

    pipeline = context.getDevice().createComputePipelineUnique({}, {{}, shaderStage, *pipelineLayout}).value;

    // Descriptor pool
    std::vector<vk::DescriptorPoolSize> poolSizes = {{vk::DescriptorType::eStorageImage, 2}};
    
    // Allocate descriptor set (store as UniqueDescriptorSet)
    const vk::DescriptorSetAllocateInfo allocInfo(context.getDescriptorPool(), 1, &descriptorSetLayout.get());
    auto descriptorSets = context.getDevice().allocateDescriptorSetsUnique(allocInfo);
    descriptorSet = std::move(descriptorSets.front()); // descriptorSet is vk::UniqueDescriptorSet

    writeDescriptors(inputImage);
}

void Tonemapper::writeDescriptors(const Image& inputImage)
{
    std::vector imageInfos = {
        vk::DescriptorImageInfo({}, inputImage.getView(), vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, outputImage.getView(), vk::ImageLayout::eGeneral)
    };

    const std::vector writes = {
        vk::WriteDescriptorSet()
        .setDstSet(descriptorSet.get())
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eStorageImage)
        .setImageInfo(imageInfos[0])
        .setDescriptorCount(1),

        vk::WriteDescriptorSet()
        .setDstSet(descriptorSet.get())
        .setDstBinding(1)
        .setDescriptorType(vk::DescriptorType::eStorageImage)
        .setImageInfo(imageInfos[1])
        .setDescriptorCount(1)
    };

    context.getDevice().updateDescriptorSets(writes, {});
}

Tonemapper::~Tonemapper()
{
    LOG_INFO("Destroying Tonemapper");
}

void Tonemapper::dispatch(const vk::CommandBuffer commandBuffer) {
    outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eGeneral);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, descriptorSet.get(), {});
    const uint32_t groupCountX = (outputImage.getWidth() + GROUP_SIZE - 1) / GROUP_SIZE;
    const uint32_t groupCountY = (outputImage.getHeight() + GROUP_SIZE - 1) / GROUP_SIZE;
    commandBuffer.dispatch(groupCountX, groupCountY, 1);
    outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Tonemapper::resize(const uint32_t width, const uint32_t height, const Image& inputImage, const vk::Format outputImageFormat)
{
    if (width == 0 || height == 0)
        return;

    context.getDevice().waitIdle();
    if (outputImage.getWidth() != width || outputImage.getHeight() != height || outputImage.getFormat() != outputImageFormat)
        outputImage = Image(context, width, height, outputImageFormat, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
    writeDescriptors(inputImage);
}
