#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

#include "internal.hpp"

#include <cstring>

namespace gpu::detail {

namespace {
vk::CullModeFlags cull_mode(const CullMode mode) {
    switch (mode) {
    case CullMode::Front: return vk::CullModeFlagBits::eFront;
    case CullMode::FrontAndBack: return vk::CullModeFlagBits::eFrontAndBack;
    case CullMode::None: return vk::CullModeFlagBits::eNone;
    default: return vk::CullModeFlagBits::eBack;
    }
}
vk::FrontFace front_face(const FrontFace face) {
    return face == FrontFace::Clockwise ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise;
}
vk::CompareOp compare_op(const CompareOp op) {
    switch (op) {
    case CompareOp::Never: return vk::CompareOp::eNever;
    case CompareOp::Less: return vk::CompareOp::eLess;
    case CompareOp::Equal: return vk::CompareOp::eEqual;
    case CompareOp::LessOrEqual: return vk::CompareOp::eLessOrEqual;
    case CompareOp::Greater: return vk::CompareOp::eGreater;
    case CompareOp::NotEqual: return vk::CompareOp::eNotEqual;
    case CompareOp::GreaterOrEqual: return vk::CompareOp::eGreaterOrEqual;
    default: return vk::CompareOp::eAlways;
    }
}
vk::PolygonMode polygon_mode(const PolygonMode mode) {
    return mode == PolygonMode::Line ? vk::PolygonMode::eLine
        : mode == PolygonMode::Point ? vk::PolygonMode::ePoint : vk::PolygonMode::eFill;
}
}

std::shared_ptr<GraphicsPipelineImpl> DeviceImpl::create_graphics(const GraphicsPipelineDesc& desc) {
    if (!desc.vertex.impl_ || !desc.fragment.impl_)
        throw Error(ErrorCode::InvalidResource, "graphics pipelines require vertex and fragment shaders");
    auto result = std::make_shared<GraphicsPipelineImpl>();
    result->device = self_.lock();
    result->state = desc.state;
    result->color_format = to_vulkan_format(desc.color_format);
    result->uses_depth = desc.state.depth_test || desc.state.depth_write;

    const std::vector<vk::PipelineShaderStageCreateInfo> stages{
        {{}, vk::ShaderStageFlagBits::eVertex, *desc.vertex.impl_->module,
         desc.vertex.impl_->entry_point.c_str()},
        {{}, vk::ShaderStageFlagBits::eFragment, *desc.fragment.impl_->module,
         desc.fragment.impl_->entry_point.c_str()}};

    const vk::PipelineVertexInputStateCreateInfo vertex_input{};
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly(
        {}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);
    const vk::PipelineViewportStateCreateInfo viewport({}, 1, nullptr, 1, nullptr);
    const vk::PipelineRasterizationStateCreateInfo raster({}, VK_FALSE, VK_FALSE,
        polygon_mode(desc.state.polygon), cull_mode(desc.state.cull),
        front_face(desc.state.front_face), VK_FALSE, 0, 0, 0, 1.0f);
    const vk::PipelineMultisampleStateCreateInfo multisample({}, vk::SampleCountFlagBits::e1);
    vk::PipelineDepthStencilStateCreateInfo depth{};
    depth.depthTestEnable = desc.state.depth_test;
    depth.depthWriteEnable = desc.state.depth_write;
    depth.depthCompareOp = compare_op(desc.state.depth_compare);
    vk::PipelineColorBlendAttachmentState blend{};
    blend.blendEnable = desc.state.blend.enabled;
    blend.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    blend.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    blend.colorBlendOp = vk::BlendOp::eAdd;
    blend.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    blend.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    blend.alphaBlendOp = vk::BlendOp::eAdd;
    blend.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
        | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    const vk::PipelineColorBlendStateCreateInfo color_blend({}, VK_FALSE, vk::LogicOp::eCopy, blend);
    const std::vector<vk::DynamicState> dynamic_states{
        vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamic({}, dynamic_states);

    const vk::PipelineCreateFlags2CreateInfo flags2{
        vk::PipelineCreateFlagBits2::eDescriptorHeapEXT};
    vk::PipelineRenderingCreateInfo rendering({}, result->color_format);
    if (result->uses_depth)
        rendering.depthAttachmentFormat = depth_format;
    rendering.pNext = &flags2;

    vk::GraphicsPipelineCreateInfo info{};
    info.setStages(stages)
        .setPVertexInputState(&vertex_input)
        .setPInputAssemblyState(&input_assembly)
        .setPViewportState(&viewport)
        .setPRasterizationState(&raster)
        .setPMultisampleState(&multisample)
        .setPDepthStencilState(&depth)
        .setPColorBlendState(&color_blend)
        .setPDynamicState(&dynamic)
        .setPNext(&rendering);
    try {
        result->pipeline = vk_device().createGraphicsPipelineUnique({}, info).value;
    } catch (const vk::SystemError& error) {
        throw Error(ErrorCode::ShaderCreationFailed, error.what());
    }
    return result;
}

void DeviceImpl::render(const RenderTarget& target, const std::function<void()>& draw_commands) {
    const auto image = find_image(target.color);
    if (!image)
        throw Error(ErrorCode::InvalidResource, "render target color image is invalid");
    const auto depth = find_image(target.depth);
    if (target.depth && !depth)
        throw Error(ErrorCode::InvalidResource, "render target depth image is invalid");
    const bool clear_attachments = target.clear;
    const bool flip_y = target.flip_y;
    submit([this, image, depth, clear_attachments, flip_y, &draw_commands]
        (const vk::CommandBuffer command) {
        const vk::AttachmentLoadOp load = clear_attachments
            ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
        const vk::ClearValue clear(std::array<float, 4>{0, 0, 0, 1});
        vk::RenderingAttachmentInfo attachment{};
        attachment.setImageView(image->view.get())
            .setImageLayout(vk::ImageLayout::eGeneral)
            .setLoadOp(load)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setClearValue(clear);
        vk::RenderingInfo rendering{};
        rendering.setRenderArea({{0, 0}, {image->width, image->height}})
            .setLayerCount(1)
            .setColorAttachments(attachment);
        vk::RenderingAttachmentInfo depth_attachment{};
        if (depth) {
            depth_attachment.setImageView(depth->view.get())
                .setImageLayout(vk::ImageLayout::eGeneral)
                .setLoadOp(load)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setClearValue(vk::ClearValue(vk::ClearDepthStencilValue{1.0f, 0}));
            rendering.setPDepthAttachment(&depth_attachment);
        }
        command.beginRendering(rendering);
        active_command_ = command;
        active_width_ = image->width;
        active_height_ = image->height;
        active_flip_y_ = flip_y;
        active_color_format_ = image->format;
        active_has_depth_ = static_cast<bool>(depth);
        // Clear the render scope on every exit path so a throwing draw
        // callback cannot leave the device believing it is still recording.
        struct ScopeExit {
            DeviceImpl& device;
            vk::CommandBuffer command;
            ~ScopeExit() {
                device.active_command_ = nullptr;
                device.active_color_format_ = vk::Format::eUndefined;
                device.active_has_depth_ = false;
                command.endRendering();
            }
        } scope{*this, command};
        draw_commands();
    });
}

void DeviceImpl::begin_draw(const GraphicsPipelineImpl& pipeline) {
    if (!active_command_)
        throw Error(ErrorCode::InvalidState,
            "GraphicsPipeline draws must be recorded inside Device::render");
    if (pipeline.color_format != active_color_format_)
        throw Error(ErrorCode::InvalidArgument,
            "graphics pipeline color format does not match the render target");
    // Dynamic rendering requires the pipeline's attachment formats to match
    // the render pass instance, depth included.
    if (pipeline.uses_depth != active_has_depth_)
        throw Error(ErrorCode::InvalidArgument,
            pipeline.uses_depth
                ? "graphics pipeline uses depth but the render target has no depth image"
                : "render target has a depth image but the graphics pipeline ignores it");
    active_command_.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline.pipeline);
    const auto width = static_cast<float>(active_width_);
    const auto height = static_cast<float>(active_height_);
    // A negative height flips Vulkan's Y-down viewport back to the Y-up NDC
    // that glm's projection matrices assume.
    active_command_.setViewport(0, active_flip_y_
        ? vk::Viewport(0, height, width, -height, 0, 1)
        : vk::Viewport(0, 0, width, height, 0, 1));
    active_command_.setScissor(0, vk::Rect2D({0, 0}, {active_width_, active_height_}));
}

void DeviceImpl::record_draw(const GraphicsPipelineImpl& pipeline, const std::uint32_t vertex_count,
    const std::uint32_t instance_count, const void* args, const std::size_t size) {
    const vk::DeviceAddress root = stage_arguments(args, size);
    begin_draw(pipeline);
    push_root(active_command_, root);
    active_command_.draw(vertex_count, instance_count, 0, 0);
}

void DeviceImpl::record_draw_indirect(const GraphicsPipelineImpl& pipeline,
    const GpuPtr<DrawArgs> commands, const std::uint32_t draw_count,
    const void* args, const std::size_t size) {
    if (!pipeline.pipeline || !commands.address || draw_count == 0)
        throw Error(ErrorCode::InvalidArgument, "invalid indirect graphics draw");
    const auto [buffer, offset] = find_buffer(commands.address);
    const vk::DeviceAddress root = stage_arguments(args, size);
    begin_draw(pipeline);
    push_root(active_command_, root);
    active_command_.drawIndirect(buffer, offset, draw_count, sizeof(DrawArgs));
}

void GraphicsPipelineImpl::draw(const std::uint32_t vertex_count, const void* args, const std::size_t size) const {
    if (!device || !pipeline)
        throw Error(ErrorCode::InvalidResource, "graphics pipeline is empty");
    device->retain_active(const_cast<GraphicsPipelineImpl*>(this)->shared_from_this());
    device->record_draw(*this, vertex_count, 1, args, size);
}

void GraphicsPipelineImpl::draw_indirect(const GpuPtr<DrawArgs> commands,
    const std::uint32_t draw_count, const void* args, const std::size_t size) const {
    if (!device || !pipeline)
        throw Error(ErrorCode::InvalidResource, "graphics pipeline is empty");
    device->retain_active(const_cast<GraphicsPipelineImpl*>(this)->shared_from_this());
    device->record_draw_indirect(*this, commands, draw_count, args, size);
}

} // namespace gpu::detail

namespace gpu {
void GraphicsPipeline::draw(const std::uint32_t vertex_count) const {
    draw_bytes(vertex_count, nullptr, 0);
}
void GraphicsPipeline::draw_bytes(const std::uint32_t vertex_count, const void* args,
    const std::size_t size) const {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "graphics pipeline is empty");
    impl_->draw(vertex_count, args, size);
}
void GraphicsPipeline::draw_instanced(const std::uint32_t vertex_count,
    const std::uint32_t instance_count) const {
    draw_instanced_bytes(vertex_count, instance_count, nullptr, 0);
}
void GraphicsPipeline::draw_instanced_bytes(const std::uint32_t vertex_count,
    const std::uint32_t instance_count, const void* args, const std::size_t size) const {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "graphics pipeline is empty");
    if (instance_count == 0)
        return;
    impl_->device->retain_active(impl_);
    impl_->device->record_draw(*impl_, vertex_count, instance_count, args, size);
}
void GraphicsPipeline::draw_indirect(const GpuPtr<DrawArgs> commands,
    const std::uint32_t draw_count) const {
    draw_indirect_bytes(commands, draw_count, nullptr, 0);
}
void GraphicsPipeline::draw_indirect_bytes(const GpuPtr<DrawArgs> commands,
    const std::uint32_t draw_count, const void* args, const std::size_t size) const {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "graphics pipeline is empty");
    impl_->draw_indirect(commands, draw_count, args, size);
}
}
