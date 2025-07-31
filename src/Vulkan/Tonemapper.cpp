#include "Tonemapper.h"
#include <iostream>
#include "Utils.h"

Tonemapper::Tonemapper(Context& context, uint32_t width, uint32_t height, const Image& inputImage)
: outputImage(context, width, height, context.chooseSwapSurfaceFormat().format,  vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc |vk::ImageUsageFlagBits::eTransferDst)
{
    //Load shader
    static constexpr unsigned char code[] = {
        #embed "../shaders/Tonemapper.spv"
    };
    shaderModule = context.getDevice().createShaderModuleUnique({{}, sizeof(code), reinterpret_cast<const uint32_t*>(code)});

    // Descriptor set layout with 2 storage images
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}
    };

    descriptorSetLayout = context.getDevice().createDescriptorSetLayoutUnique({{}, static_cast<uint32_t>(bindings.size()), bindings.data()});

    // Pipeline layout (no push constants)
    pipelineLayout = context.getDevice().createPipelineLayoutUnique({{}, 1, &*descriptorSetLayout});

    // Compute pipeline
    vk::PipelineShaderStageCreateInfo shaderStage({}, vk::ShaderStageFlagBits::eCompute, *shaderModule, "main");

    pipeline = context.getDevice().createComputePipelineUnique({}, {{}, shaderStage, *pipelineLayout}).value;

    // Descriptor pool
    std::vector<vk::DescriptorPoolSize> poolSizes = {{vk::DescriptorType::eStorageImage, 2}};
    
    // Allocate descriptor set (store as UniqueDescriptorSet)
    vk::DescriptorSetAllocateInfo allocInfo(context.getDescriptorPool(), 1, &descriptorSetLayout.get());
    auto descriptorSets = context.getDevice().allocateDescriptorSetsUnique(allocInfo);
    descriptorSet = std::move(descriptorSets.front()); // descriptorSet is vk::UniqueDescriptorSet

    // Write descriptors
    std::vector<vk::DescriptorImageInfo> imageInfos = {
        vk::DescriptorImageInfo({}, inputImage.getImageView(), vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, outputImage.getImageView(), vk::ImageLayout::eGeneral)
    };

    std::vector<vk::WriteDescriptorSet> writes = {
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
    std::cout << "Destroying Tonemapper" << std::endl;
}

void Tonemapper::dispatch(vk::CommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z) {
    outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eGeneral);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, descriptorSet.get(), {});
    commandBuffer.dispatch(x, y, z);
    outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal);
}
