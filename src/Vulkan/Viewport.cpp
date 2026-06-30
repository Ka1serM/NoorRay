#include "Viewport.h"
#include <iostream>

#include "Globals.h"
#include "Log.h"
#include "ViewportSpv.h"

namespace
{
constexpr uint32_t ViewportGroupSize = 16;

struct ViewportPushConstants
{
    uint32_t selectedIndex;
    float    exposure;
    int32_t  bufferVisualization;
    int32_t  tonemappingEnabled;
};
}

Viewport::Viewport(Context& context, const uint32_t width, const uint32_t height,
                   const Image& color0,    const Image& color1,
                   const Image& albedo0,   const Image& albedo1,
                   const Image& normal0,   const Image& normal1,
                   const Image& crypto0,   const Image& crypto1,
                   const Image& position0, const Image& position1,
                   const vk::Format outputImageFormat)
: context(context),
  outputImage(context, width, height, outputImageFormat,
      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
      vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst)
{
    shaderModule = context.getDevice().createShaderModuleUnique(
        {{}, noorRayViewportSpvLength, reinterpret_cast<const uint32_t*>(noorRayViewportSpv)});

    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}, // color
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}, // output
        {2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}, // cryptomatte
        {3, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}, // albedo
        {4, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}, // normal
        {5, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}, // position
    };

    descriptorSetLayout = context.getDevice().createDescriptorSetLayoutUnique(
        {{}, static_cast<uint32_t>(bindings.size()), bindings.data()});

    const vk::PushConstantRange pushConstantRange(
        vk::ShaderStageFlagBits::eCompute, 0, sizeof(ViewportPushConstants));
    pipelineLayout = context.getDevice().createPipelineLayoutUnique(
        {{}, 1, &*descriptorSetLayout, 1, &pushConstantRange});

    vk::PipelineShaderStageCreateInfo shaderStage({}, vk::ShaderStageFlagBits::eCompute, *shaderModule, "main");
    pipeline = context.getDevice().createComputePipelineUnique({}, {{}, shaderStage, *pipelineLayout}).value;

    const std::array layouts{descriptorSetLayout.get(), descriptorSetLayout.get()};
    const vk::DescriptorSetAllocateInfo allocInfo(context.getDescriptorPool(), 2, layouts.data());
    auto descriptorSets = context.getDevice().allocateDescriptorSetsUnique(allocInfo);
    this->descriptorSets[0] = std::move(descriptorSets[0]);
    this->descriptorSets[1] = std::move(descriptorSets[1]);

    writeDescriptors(0, color0, albedo0, normal0, crypto0, position0);
    writeDescriptors(1, color1, albedo1, normal1, crypto1, position1);
}

void Viewport::writeDescriptors(const uint32_t bufferIndex,
    const Image& color, const Image& albedo, const Image& normal,
    const Image& crypto, const Image& position)
{
    std::vector imageInfos = {
        vk::DescriptorImageInfo({}, color.getView(),      vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, outputImage.getView(),vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, crypto.getView(),     vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, albedo.getView(),     vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, normal.getView(),     vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, position.getView(),   vk::ImageLayout::eGeneral),
    };

    std::vector<vk::WriteDescriptorSet> writes;
    for (uint32_t i = 0; i < static_cast<uint32_t>(imageInfos.size()); ++i)
    {
        writes.push_back(vk::WriteDescriptorSet()
            .setDstSet(descriptorSets[bufferIndex].get())
            .setDstBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageImage)
            .setImageInfo(imageInfos[i])
            .setDescriptorCount(1));
    }

    context.getDevice().updateDescriptorSets(writes, {});
}

Viewport::~Viewport()
{
    LOG_INFO("Destroying Viewport");
}

void Viewport::dispatch(
    const vk::CommandBuffer commandBuffer,
    const uint32_t bufferIndex,
    const uint32_t selectedIndex,
    const float exposure,
    const int bufferVisualization,
    const int tonemappingEnabled)
{
    outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eGeneral);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0,
                                     descriptorSets[bufferIndex].get(), {});
    const ViewportPushConstants pushConstants{
        selectedIndex, exposure, bufferVisualization, tonemappingEnabled};
    commandBuffer.pushConstants(
        *pipelineLayout, vk::ShaderStageFlagBits::eCompute,
        0, sizeof(pushConstants), &pushConstants);
    const uint32_t groupCountX = (outputImage.getWidth()  + ViewportGroupSize - 1) / ViewportGroupSize;
    const uint32_t groupCountY = (outputImage.getHeight() + ViewportGroupSize - 1) / ViewportGroupSize;
    commandBuffer.dispatch(groupCountX, groupCountY, 1);
    outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Viewport::resize(const uint32_t width, const uint32_t height,
                      const Image& color0,    const Image& color1,
                      const Image& albedo0,   const Image& albedo1,
                      const Image& normal0,   const Image& normal1,
                      const Image& crypto0,   const Image& crypto1,
                      const Image& position0, const Image& position1,
                      const vk::Format outputImageFormat)
{
    if (width == 0 || height == 0)
        return;

    context.getDevice().waitIdle();
    if (outputImage.getWidth() != width || outputImage.getHeight() != height ||
        outputImage.getFormat() != outputImageFormat)
    {
        outputImage = Image(context, width, height, outputImageFormat,
            vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
    }
    writeDescriptors(0, color0, albedo0, normal0, crypto0, position0);
    writeDescriptors(1, color1, albedo1, normal1, crypto1, position1);
}
