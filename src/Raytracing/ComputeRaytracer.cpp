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
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}, // Mesh instances buffer
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}, // Output color image
        {2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}, // Output albedo image
        {3, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}, // Output normal image
        {4, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}, // Output crypto  image
        {5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}, // Mesh buffer
        {6, vk::DescriptorType::eCombinedImageSampler, MAX_TEXTURES, vk::ShaderStageFlagBits::eCompute}, // Textures
    };

    createDescriptorSet(bindings);

    // 4. Create the pipeline layout
    vk::PushConstantRange pushRange{};
    pushRange.setOffset(0);
    pushRange.setSize(sizeof(PushConstantsData));
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
    
    bindOutputImages();    
}


void ComputeRaytracer::updateTLAS()
{
    std::vector<ComputeInstance> instances;

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

    if (instances.empty())
    {
        auto emptyInstance = ComputeInstance{};
        emptyInstance.meshId = UINT32_MAX; // invalid instance        instancesBuffer = {context, Buffer::Type::Storage, sizeof(ComputeInstance), &emptyInstance};
    }
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


void ComputeRaytracer::render(const vk::CommandBuffer& commandBuffer, const PushConstantsData& pushConstants)
{
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.get());
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, descriptorSet.get(), {});
    commandBuffer.pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstantsData), &pushConstants);

    uint32_t groupCountX = (width + GROUP_SIZE - 1) / GROUP_SIZE;
    uint32_t groupCountY = (height + GROUP_SIZE - 1) / GROUP_SIZE;
    commandBuffer.dispatch(groupCountX, groupCountY, 1);
}
