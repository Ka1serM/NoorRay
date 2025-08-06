#include "ComputeRaytracer.h"

#include "Globals.h"
#include "Utils.h"
#include "Scene/MeshInstance.h"

ComputeRaytracer::ComputeRaytracer(Scene& scene, uint32_t width, uint32_t height)
    : Raytracer(scene, width, height)
{
    static constexpr unsigned char ComputeShader[] = {
        #embed "../shaders/PathTracing/PathTracer.spv"
    };

    vk::UniqueShaderModule computeShaderModule = context.getDevice().createShaderModuleUnique({{}, sizeof(ComputeShader), reinterpret_cast<const uint32_t*>(ComputeShader)});

    vk::PipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.setStage(vk::ShaderStageFlagBits::eCompute);
    shaderStageInfo.setModule(*computeShaderModule);
    shaderStageInfo.setPName("main");

    // Define the descriptor set layout bindings.
    std::vector<vk::DescriptorSetLayoutBinding> bindings{
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {3, vk::DescriptorType::eCombinedImageSampler, MAX_TEXTURES, vk::ShaderStageFlagBits::eCompute}
    };

    std::vector<vk::DescriptorBindingFlags> bindingFlags(bindings.size(), vk::DescriptorBindingFlags{});
    bindingFlags[3] = vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind;

    vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.setBindingFlags(bindingFlags);

    vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo{};
    descSetLayoutInfo.setBindings(bindings);
    descSetLayoutInfo.setPNext(&bindingFlagsInfo);
    // This flag is required for the entire set when using eUpdateAfterBind
    descSetLayoutInfo.setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool); 

    descSetLayout = context.getDevice().createDescriptorSetLayoutUnique(descSetLayoutInfo);
    
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(context.getDescriptorPool());
    allocInfo.setSetLayouts(descSetLayout.get());
    allocInfo.setDescriptorSetCount(1);
    descriptorSet = std::move(context.getDevice().allocateDescriptorSetsUnique(allocInfo).front());

    // 4. Create the pipeline layout
    vk::PushConstantRange pushRange{};
    pushRange.setOffset(0);
    pushRange.setSize(sizeof(PushConstants));
    pushRange.setStageFlags(vk::ShaderStageFlagBits::eCompute);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setSetLayouts(descSetLayout.get());
    pipelineLayoutInfo.setPushConstantRanges(pushRange);
    pipelineLayout = context.getDevice().createPipelineLayoutUnique(pipelineLayoutInfo);

    // 5. Create the compute pipeline
    vk::ComputePipelineCreateInfo computePipelineInfo{};
    computePipelineInfo.setStage(shaderStageInfo);
    computePipelineInfo.setLayout(pipelineLayout.get());

    auto pipelineResult = context.getDevice().createComputePipelineUnique({}, computePipelineInfo);
    if (pipelineResult.result != vk::Result::eSuccess)
        throw std::runtime_error("failed to create compute pipeline.");

    pipeline = std::move(pipelineResult.value);

    // 6. Update descriptor set with the output image
    vk::DescriptorImageInfo storageImageInfo{};
    storageImageInfo.setImageView(outputImage.getImageView());
    storageImageInfo.setImageLayout(vk::ImageLayout::eGeneral);

    vk::WriteDescriptorSet storageImageWrite{};
    storageImageWrite.setDstSet(descriptorSet.get());
    storageImageWrite.setDstBinding(1);
    storageImageWrite.setDescriptorType(vk::DescriptorType::eStorageImage);
    storageImageWrite.setDescriptorCount(1);
    storageImageWrite.setImageInfo(storageImageInfo);
    context.getDevice().updateDescriptorSets(storageImageWrite, {});
}


void ComputeRaytracer::updateTLAS()
{
    std::vector<ComputeInstance> instances;
    {
        auto lock = scene.shared_lock();
        const auto& meshInstances = scene.getMeshInstances();
        instances.reserve(meshInstances.size());

        for (const auto* meshInstance : meshInstances)
        {
            const mat4 transform = meshInstance->getTransform().getMatrix();
            instances.push_back({
                .transform = transform,
                .inverseTransform = inverse(transform),
                .meshId = meshInstance->getMeshAsset().getMeshIndex()
            });
        }
    }

    if (instances.empty())
        instancesBuffer = Buffer();
    else
        instancesBuffer = {context, Buffer::Type::Storage, sizeof(ComputeInstance) * instances.size(), instances.data()};

    vk::DescriptorBufferInfo bufferInfo = instancesBuffer.getDescriptorInfo();
    vk::WriteDescriptorSet write{};
    write.setDstSet(descriptorSet.get());
    write.setDstBinding(0);
    write.setDescriptorType(vk::DescriptorType::eStorageBuffer);
    write.setDescriptorCount(1);
    write.setBufferInfo(bufferInfo);
    context.getDevice().updateDescriptorSets(write, {});
}


void ComputeRaytracer::render(const vk::CommandBuffer& commandBuffer, const PushConstants& pushConstants)
{
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.get());
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, descriptorSet.get(), {});
    commandBuffer.pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants), &pushConstants);

    uint32_t groupCountX = (width + GROUP_SIZE - 1) / GROUP_SIZE;
    uint32_t groupCountY = (height + GROUP_SIZE - 1) / GROUP_SIZE;
    commandBuffer.dispatch(groupCountX, groupCountY, 1);
}
