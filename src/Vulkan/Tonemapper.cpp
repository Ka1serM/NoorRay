#include "Tonemapper.h"
#include <iostream>

#include "Globals.h"
#include "Log.h"
#include "ViewportSpv.h"

namespace
{
constexpr uint32_t ViewportGroupSize = 16;
}

Tonemapper::Tonemapper(Context& context, const uint32_t width, const uint32_t height,
                       const Image& inputImage0, const Image& inputImage1,
                       const Image& idImage0, const Image& idImage1,
                       const vk::Format outputImageFormat)
: context(context),
  outputImage(context, width, height, outputImageFormat,  vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc |vk::ImageUsageFlagBits::eTransferDst)
{
    //Load shader
    shaderModule = context.getDevice().createShaderModuleUnique(
        {{}, noorRayViewportSpvLength, reinterpret_cast<const uint32_t*>(noorRayViewportSpv)});

    // Descriptor set layout with color input, output, and object ID input.
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}
    };

    descriptorSetLayout = context.getDevice().createDescriptorSetLayoutUnique({{}, static_cast<uint32_t>(bindings.size()), bindings.data()});

    const vk::PushConstantRange pushConstantRange(
        vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t));
    pipelineLayout = context.getDevice().createPipelineLayoutUnique(
        {{}, 1, &*descriptorSetLayout, 1, &pushConstantRange});

    // Compute pipeline
    vk::PipelineShaderStageCreateInfo shaderStage({}, vk::ShaderStageFlagBits::eCompute, *shaderModule, "main");

    pipeline = context.getDevice().createComputePipelineUnique({}, {{}, shaderStage, *pipelineLayout}).value;

    // Allocate descriptor set (store as UniqueDescriptorSet)
    const std::array layouts{descriptorSetLayout.get(), descriptorSetLayout.get()};
    const vk::DescriptorSetAllocateInfo allocInfo(context.getDescriptorPool(), 2, layouts.data());
    auto descriptorSets = context.getDevice().allocateDescriptorSetsUnique(allocInfo);
    this->descriptorSets[0] = std::move(descriptorSets[0]);
    this->descriptorSets[1] = std::move(descriptorSets[1]);

    writeDescriptors(0, inputImage0, idImage0);
    writeDescriptors(1, inputImage1, idImage1);
}

void Tonemapper::writeDescriptors(
    const uint32_t bufferIndex, const Image& inputImage, const Image& idImage)
{
    std::vector imageInfos = {
        vk::DescriptorImageInfo({}, inputImage.getView(), vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, outputImage.getView(), vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, idImage.getView(), vk::ImageLayout::eGeneral)
    };

    const std::vector writes = {
        vk::WriteDescriptorSet()
        .setDstSet(descriptorSets[bufferIndex].get())
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eStorageImage)
        .setImageInfo(imageInfos[0])
        .setDescriptorCount(1),

        vk::WriteDescriptorSet()
        .setDstSet(descriptorSets[bufferIndex].get())
        .setDstBinding(1)
        .setDescriptorType(vk::DescriptorType::eStorageImage)
        .setImageInfo(imageInfos[1])
        .setDescriptorCount(1),

        vk::WriteDescriptorSet()
        .setDstSet(descriptorSets[bufferIndex].get())
        .setDstBinding(2)
        .setDescriptorType(vk::DescriptorType::eStorageImage)
        .setImageInfo(imageInfos[2])
        .setDescriptorCount(1)
    };

    context.getDevice().updateDescriptorSets(writes, {});
}

Tonemapper::~Tonemapper()
{
    LOG_INFO("Destroying Tonemapper");
}

void Tonemapper::dispatch(
    const vk::CommandBuffer commandBuffer,
    const uint32_t bufferIndex,
    const uint32_t selectedIndex)
{
    outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eGeneral);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0,
                                     descriptorSets[bufferIndex].get(), {});
    commandBuffer.pushConstants(
        *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(selectedIndex), &selectedIndex);
    const uint32_t groupCountX = (outputImage.getWidth() + ViewportGroupSize - 1) / ViewportGroupSize;
    const uint32_t groupCountY = (outputImage.getHeight() + ViewportGroupSize - 1) / ViewportGroupSize;
    commandBuffer.dispatch(groupCountX, groupCountY, 1);
    outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Tonemapper::resize(const uint32_t width, const uint32_t height,
                        const Image& inputImage0, const Image& inputImage1,
                        const Image& idImage0, const Image& idImage1,
                        const vk::Format outputImageFormat)
{
    if (width == 0 || height == 0)
        return;

    context.getDevice().waitIdle();
    if (outputImage.getWidth() != width || outputImage.getHeight() != height || outputImage.getFormat() != outputImageFormat)
        outputImage = Image(context, width, height, outputImageFormat, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
    writeDescriptors(0, inputImage0, idImage0);
    writeDescriptors(1, inputImage1, idImage1);
}
