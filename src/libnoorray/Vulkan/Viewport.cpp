#include "Viewport.h"
#include <iostream>

#include "Globals.h"
#include "Log.h"
#include "Scene/Scene.h"
#include "Scene/LightInstance.h"

namespace
{
alignas(uint32_t) constexpr unsigned char noorRayViewportSpv[] = {
    #embed "../Shaders/Viewport/Viewport.spv"
};
constexpr std::size_t noorRayViewportSpvLength = sizeof(noorRayViewportSpv);

alignas(uint32_t) constexpr unsigned char noorRayViewportBillboardsSpv[] = {
    #embed "../Shaders/Viewport/ViewportBillboards.spv"
};
constexpr std::size_t noorRayViewportBillboardsSpvLength = sizeof(noorRayViewportBillboardsSpv);

constexpr uint32_t ViewportGroupSize = 16;

struct ViewportPushConstants
{
    uint32_t selectedCryptomatteId;
    float    exposure;
    int32_t  bufferVisualization;
    int32_t  tonemappingEnabled;
};

struct BillboardPushConstants
{
    glm::mat4 viewProjection;
    glm::vec2 screenSize;
    float     radius;
};
}

Viewport::Viewport(Context& context, const uint32_t width, const uint32_t height,
                   const Image& color,    const Image& albedo,
                   const Image& normal,   const Image& crypto,
                   const Image& position,
                   const vk::Format outputImageFormat)
: context(context),
  outputImage(context, width, height, outputImageFormat,
      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
      vk::ImageUsageFlagBits::eColorAttachment |
      vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst),
  billboards(context)
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

    const std::array layouts{descriptorSetLayout.get()};
    const vk::DescriptorSetAllocateInfo allocInfo(context.getDescriptorPool(), 1, layouts.data());
    auto descriptorSets = context.getDevice().allocateDescriptorSetsUnique(allocInfo);
    this->descriptorSets[0] = std::move(descriptorSets[0]);

    writeDescriptors(color, albedo, normal, crypto, position);

    createBillboardPipeline();
    reserveBillboards(1);
}

void Viewport::writeDescriptors(
    const Image& color, const Image& albedo, const Image& normal,
    const Image& crypto, const Image& position)
{
    // Every binding is a storage image the compute pass reads or writes, so a
    // set is only usable once all of them exist. Before the first resize, or
    // with AOVs switched off, some do not — leave the set unwritten and let
    // dispatch() skip until a later call supplies the full complement.
    descriptorsValid = color.getView() && outputImage.getView()
        && crypto.getView() && albedo.getView() && normal.getView()
        && position.getView();
    if (!descriptorsValid)
        return;

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
            .setDstSet(descriptorSets[0].get())
            .setDstBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageImage)
            .setImageInfo(imageInfos[i])
            .setDescriptorCount(1));
    }

    context.getDevice().updateDescriptorSets(writes, {});
}

void Viewport::createBillboardPipeline()
{
    billboardShaderModule = context.getDevice().createShaderModuleUnique(
        {{}, noorRayViewportBillboardsSpvLength, reinterpret_cast<const uint32_t*>(noorRayViewportBillboardsSpv)});

    const vk::DescriptorSetLayoutBinding billboardBinding{
        0, vk::DescriptorType::eStorageBuffer, 1,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment};
    billboardDescriptorSetLayout = context.getDevice().createDescriptorSetLayoutUnique(
        {{}, 1, &billboardBinding});

    const vk::PushConstantRange pushConstantRange(
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        0, sizeof(BillboardPushConstants));
    billboardPipelineLayout = context.getDevice().createPipelineLayoutUnique(
        {{}, 1, &*billboardDescriptorSetLayout, 1, &pushConstantRange});

    const vk::DescriptorSetAllocateInfo allocInfo(context.getDescriptorPool(), 1, &*billboardDescriptorSetLayout);
    billboardDescriptorSet = std::move(context.getDevice().allocateDescriptorSetsUnique(allocInfo).front());

    const std::array stages{
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, *billboardShaderModule, "vertMain"),
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, *billboardShaderModule, "fragMain"),
    };

    // No vertex buffers — the vertex shader builds a screen-facing quad from
    // SV_VertexID/SV_InstanceID and reads billboard data straight out of the
    // storage buffer.
    constexpr vk::PipelineVertexInputStateCreateInfo vertexInput{};
    constexpr vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
        {}, vk::PrimitiveTopology::eTriangleStrip, false);

    constexpr vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1, nullptr);

    vk::PipelineRasterizationStateCreateInfo rasterization{};
    rasterization.polygonMode = vk::PolygonMode::eFill;
    rasterization.cullMode = vk::CullModeFlagBits::eNone;
    rasterization.lineWidth = 1.0f;

    vk::PipelineMultisampleStateCreateInfo multisample{};
    multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = true;
    blendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    blendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    blendAttachment.colorBlendOp = vk::BlendOp::eAdd;
    blendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    blendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    blendAttachment.alphaBlendOp = vk::BlendOp::eAdd;
    blendAttachment.colorWriteMask =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    const vk::PipelineColorBlendStateCreateInfo colorBlend({}, false, vk::LogicOp::eCopy, 1, &blendAttachment);

    const std::array dynamicStates{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates);

    const vk::Format colorFormat = outputImage.getFormat();
    vk::PipelineRenderingCreateInfo renderingCreateInfo{};
    renderingCreateInfo.colorAttachmentCount = 1;
    renderingCreateInfo.pColorAttachmentFormats = &colorFormat;

    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.pNext = &renderingCreateInfo;
    pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterization;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = *billboardPipelineLayout;

    billboardPipeline = context.getDevice().createGraphicsPipelineUnique({}, pipelineInfo).value;
}

void Viewport::reserveBillboards(const uint32_t capacity)
{
    if (capacity <= billboards.capacity())
        return;

    // The descriptor set may still be referenced by an in-flight command buffer from
    // a previous frame; growth is rare (only when the light count exceeds the current
    // capacity), so a wait here is cheap insurance against rewriting a live binding.
    if (billboards.capacity() > 0)
        context.getDevice().waitIdle();

    billboards.reserve(capacity);
    writeBillboardDescriptor();
}

void Viewport::writeBillboardDescriptor()
{
    const vk::WriteDescriptorSet write = vk::WriteDescriptorSet()
        .setDstSet(billboardDescriptorSet.get())
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setBufferInfo(billboards.buffer().descriptorInfo())
        .setDescriptorCount(1);
    context.getDevice().updateDescriptorSets(write, {});
}

void Viewport::updateBillboards(const Scene& scene)
{
    if (observedLightRevision == scene.getLightRevision())
        return;

    const uint32_t lightCount = scene.getPointLightCount()
        + scene.getSpotLightCount()
        + scene.getRectLightCount()
        + scene.getDirectionalLightCount();
    reserveBillboards(std::max(1u, lightCount));
    billboards.clear();
    for (const auto& obj : scene.getSceneObjects())
    {
        if (const auto* light = dynamic_cast<LightInstance*>(obj.get()))
        {
            billboards.push_back(ViewportBillboard{
                glm::vec4(light->getWorldTransform().getPosition(), static_cast<float>(light->lightType)),
                glm::vec4(light->getColor(), 1.0f)});
        }
    }
    billboardCount = static_cast<uint32_t>(billboards.size());
    observedLightRevision = scene.getLightRevision();
}

Viewport::~Viewport()
{
    LOG_INFO("Destroying Viewport");
}

void Viewport::drawBillboards(const vk::CommandBuffer commandBuffer, const glm::mat4& viewProjection)
{
    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment.setImageView(outputImage.getView());
    colorAttachment.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
    colorAttachment.setLoadOp(vk::AttachmentLoadOp::eLoad);
    colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);

    vk::RenderingInfo renderingInfo{};
    renderingInfo.setRenderArea(vk::Rect2D({0, 0}, {outputImage.getWidth(), outputImage.getHeight()}));
    renderingInfo.setLayerCount(1);
    renderingInfo.setColorAttachments(colorAttachment);

    commandBuffer.beginRendering(renderingInfo);

    // Negative height flips Vulkan's Y-down viewport convention back to the Y-up
    // NDC that glm::perspective() (used for the camera matrix) assumes.
    commandBuffer.setViewport(0, vk::Viewport(
        0.0f, static_cast<float>(outputImage.getHeight()),
        static_cast<float>(outputImage.getWidth()), -static_cast<float>(outputImage.getHeight()), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D({0, 0}, {outputImage.getWidth(), outputImage.getHeight()}));

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *billboardPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *billboardPipelineLayout, 0,
                                     billboardDescriptorSet.get(), {});

    const BillboardPushConstants pushConstants{
        // Slang emits this push-constant matrix with row-major storage. GLM
        // stores matrices column-major, so transpose once at the ABI boundary
        // to preserve the same mathematical matrix in the shader.
        glm::transpose(viewProjection),
        glm::vec2(static_cast<float>(outputImage.getWidth()), static_cast<float>(outputImage.getHeight())),
        ViewportBillboardPixelRadius};
    commandBuffer.pushConstants(
        *billboardPipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        0, sizeof(pushConstants), &pushConstants);

    commandBuffer.draw(4, billboardCount, 0, 0);

    commandBuffer.endRendering();
}

void Viewport::dispatch(
    const vk::CommandBuffer commandBuffer,
    const uint32_t selectedCryptomatteId,
    const glm::mat4& viewProjection,
    const float exposure,
    const int bufferVisualization,
    const bool tonemappingEnabled,
    const bool showBillboards)
{
    if (!descriptorsValid)
        return;

    outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eGeneral);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0,
                                     descriptorSets[0].get(), {});
    const ViewportPushConstants pushConstants{
        selectedCryptomatteId, exposure, bufferVisualization, tonemappingEnabled ? 1 : 0};
    commandBuffer.pushConstants(
        *pipelineLayout, vk::ShaderStageFlagBits::eCompute,
        0, sizeof(pushConstants), &pushConstants);
    const uint32_t groupCountX = (outputImage.getWidth()  + ViewportGroupSize - 1) / ViewportGroupSize;
    const uint32_t groupCountY = (outputImage.getHeight() + ViewportGroupSize - 1) / ViewportGroupSize;
    commandBuffer.dispatch(groupCountX, groupCountY, 1);

    if (showBillboards && billboardCount > 0)
    {
        outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eColorAttachmentOptimal);
        drawBillboards(commandBuffer, viewProjection);
    }
    outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Viewport::resize(const uint32_t width, const uint32_t height,
                      const Image& color,    const Image& albedo,
                      const Image& normal,   const Image& crypto,
                      const Image& position,
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
            vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
    }
    writeDescriptors(color, albedo, normal, crypto, position);
}
