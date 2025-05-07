#include "ConvertImageComputeShader.h"
#include "Utils.h"

ConvertImageComputeShader::ConvertImageComputeShader(const std::string& shaderPath, Context& context, vk::ImageView inputImageView, vk::ImageView outputImageView)
: context(context)
{
    // Load shader
    auto code = Utils::readFile(shaderPath);
    shaderModule = context.device->createShaderModuleUnique({ {}, code.size(), reinterpret_cast<const uint32_t*>(code.data()) });

    vk::PushConstantRange pushConstantRange(
    vk::ShaderStageFlagBits::eCompute,  // Shader stage
    0,                                  // Offset
    sizeof(int) * 2                     // Size (2 integers)
);

    // Descriptor set layout with 2 storage images
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        { 0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute },
        { 1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute }
    };

    descriptorSetLayout = context.device->createDescriptorSetLayoutUnique({ {}, static_cast<uint32_t>(bindings.size()), bindings.data() });

    // Pipeline layout
    //pipelineLayout = context.device->createPipelineLayoutUnique({ {}, 1, &*descriptorSetLayout });
    pipelineLayout = context.device->createPipelineLayoutUnique({{}, 1, &*descriptorSetLayout, 1, &pushConstantRange});

    // Compute pipeline
    vk::PipelineShaderStageCreateInfo shaderStage({}, vk::ShaderStageFlagBits::eCompute, *shaderModule, "main");

    pipeline = context.device->createComputePipelineUnique({}, { {}, shaderStage, *pipelineLayout }).value;

    // Descriptor pool
    std::vector<vk::DescriptorPoolSize> poolSizes = {{ vk::DescriptorType::eStorageImage, 2 }};

    descriptorPool = context.device->createDescriptorPool({ {}, 1, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });

    // Descriptor set allocation
    vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, 1, &*descriptorSetLayout);
    descriptorSet = context.device->allocateDescriptorSets(allocInfo)[0];

    // Write descriptors
    std::vector<vk::DescriptorImageInfo> imageInfos = {
        vk::DescriptorImageInfo({}, inputImageView, vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, outputImageView, vk::ImageLayout::eGeneral)
    };
    std::vector<vk::WriteDescriptorSet> writes = {
        { descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageImage, &imageInfos[0], nullptr, nullptr },
        { descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageImage, &imageInfos[1], nullptr, nullptr }
    };
    context.device->updateDescriptorSets(writes, {});
}

void ConvertImageComputeShader::dispatch(vk::CommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z) {
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, descriptorSet, {});
    commandBuffer.dispatch(x, y, z);
}