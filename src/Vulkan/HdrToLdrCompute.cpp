#include "HdrToLdrCompute.h"
#include "Utils.h"

HdrToLdrCompute::HdrToLdrCompute(Context& context, uint32_t width, uint32_t height, vk::ImageView inputImageView)
: outputImage(context, width, height, vk::Format::eB8G8R8A8Unorm,vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc |vk::ImageUsageFlagBits::eTransferDst)
{
    //Load shader
    static constexpr unsigned char code[] = {
        #embed "../shaders/HdrToLdrCompute.spv"
    };
    shaderModule = context.device.get().createShaderModuleUnique({
        {}, sizeof(code), reinterpret_cast<const uint32_t*>(code)
    });

    // Descriptor set layout with 2 storage images
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}
    };

    descriptorSetLayout = context.device.get().createDescriptorSetLayoutUnique({
        {}, static_cast<uint32_t>(bindings.size()), bindings.data()
    });

    // Pipeline layout (no push constants)
    pipelineLayout = context.device.get().createPipelineLayoutUnique({{}, 1, &*descriptorSetLayout});

    // Compute pipeline
    vk::PipelineShaderStageCreateInfo shaderStage({}, vk::ShaderStageFlagBits::eCompute, *shaderModule, "main");

    pipeline = context.device.get().createComputePipelineUnique({}, {{}, shaderStage, *pipelineLayout}).value;

    // Descriptor pool
    std::vector<vk::DescriptorPoolSize> poolSizes = {{vk::DescriptorType::eStorageImage, 2}};

    descriptorPool = context.device.get().createDescriptorPoolUnique({
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        1,
        static_cast<uint32_t>(poolSizes.size()),
        poolSizes.data()
    });

    // Allocate descriptor set (store as UniqueDescriptorSet)
    vk::DescriptorSetAllocateInfo allocInfo(descriptorPool.get(), 1, &descriptorSetLayout.get());
    auto descriptorSets = context.device.get().allocateDescriptorSetsUnique(allocInfo);
    descriptorSet = std::move(descriptorSets.front()); // descriptorSet is vk::UniqueDescriptorSet

    // Write descriptors
    std::vector<vk::DescriptorImageInfo> imageInfos = {
        vk::DescriptorImageInfo({}, inputImageView, vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, outputImage.view.get(), vk::ImageLayout::eGeneral)
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

    context.device.get().updateDescriptorSets(writes, {});
}

void HdrToLdrCompute::dispatch(vk::CommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z) {
    outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eGeneral);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, descriptorSet.get(), {});
    commandBuffer.dispatch(x, y, z);
    outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal);
}
