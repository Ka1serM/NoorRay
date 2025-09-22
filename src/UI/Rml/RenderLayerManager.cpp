#include "RenderLayerManager.h"

#include <cmath>
#include "RmlRenderInterface.h"
#include <RmlUi/Core/Log.h>

#include "UniformBuffer.h"
#include "RmlUi/Core/Dictionary.h"

// --- RenderLayer Implementation ---
RenderLayerManager::RenderLayer::RenderLayer(VmaAllocator allocator, vk::Device device, vk::Extent2D extent, vk::Format format,
            vk::DescriptorPool descriptor_pool, vk::DescriptorSetLayout texture_layout, vk::Sampler sampler) : extent(extent)
{
    // Create Image
    const auto image_ci = static_cast<VkImageCreateInfo>(vk::ImageCreateInfo({}, vk::ImageType::e2D, format, {extent.width, extent.height, 1}, 1, 1, 
        vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc));
    constexpr VmaAllocationCreateInfo alloc_ci = {0, VMA_MEMORY_USAGE_GPU_ONLY};
    vmaCreateImage(allocator, &image_ci, &alloc_ci, reinterpret_cast<VkImage*>(&image), &allocation, nullptr);
    
    // Create ImageView
    const vk::ImageViewCreateInfo view_ci({}, image, vk::ImageViewType::e2D, format, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    image_view = device.createImageViewUnique(view_ci);
    
    // Create Descriptor Set
    const vk::DescriptorSetAllocateInfo ds_alloc_info(descriptor_pool, texture_layout);
    texture_descriptor_set = device.allocateDescriptorSets(ds_alloc_info)[0];
    vk::DescriptorImageInfo image_info(sampler, image_view.get(), vk::ImageLayout::eShaderReadOnlyOptimal);
    vk::WriteDescriptorSet write(texture_descriptor_set, 0, 0, vk::DescriptorType::eCombinedImageSampler, image_info);
    device.updateDescriptorSets(write, {});
}


// --- RenderLayerManager Implementation ---
RenderLayerManager::RenderLayerManager(VmaAllocator allocator, vk::Device device, vk::DescriptorPool descriptor_pool,
    vk::DescriptorSetLayout texture_layout, vk::DescriptorSetLayout ubo_layout, vk::Sampler sampler,
    Rml::CompiledGeometryHandle fullscreen_quad, vk::Format color_format, vk::Format depth_format, vk::Extent2D initial_extent)
    : m_allocator(allocator), m_device(device), m_descriptor_pool(descriptor_pool), m_texture_layout(texture_layout),
      m_ubo_layout(ubo_layout), m_sampler(sampler), m_fullscreen_quad(fullscreen_quad), m_target_extent(initial_extent)
{
    CreatePipelines(color_format, depth_format);
    CreatePostProcessResources(initial_extent);
}

RenderLayerManager::~RenderLayerManager() {
    // Unique handles and RAII manage Vulkan resources.
    // The destruction map is cleared automatically.
}

void RenderLayerManager::CreatePostProcessResources(vk::Extent2D extent) {
    constexpr vk::Format layer_format = vk::Format::eR8G8B8A8Unorm;
    m_postprocess_buffers[0] = std::make_unique<RenderLayer>(m_allocator, m_device, extent, layer_format, m_descriptor_pool, m_texture_layout, m_sampler);
    m_postprocess_buffers[1] = std::make_unique<RenderLayer>(m_allocator, m_device, extent, layer_format, m_descriptor_pool, m_texture_layout, m_sampler);
}

void RenderLayerManager::CreatePipelines(vk::Format color_format, vk::Format depth_format) {
    // --- 1. Load Shader Modules ---
    m_vert_passthrough = m_device.createShaderModuleUnique({{}, sizeof(shader_passthrough_vert), reinterpret_cast<const uint32_t*>(shader_passthrough_vert)});
    m_frag_passthrough = m_device.createShaderModuleUnique({{}, sizeof(shader_passthrough_frag), reinterpret_cast<const uint32_t*>(shader_passthrough_frag)});
    m_frag_shader_filter = m_device.createShaderModuleUnique({{}, sizeof(shader_frag_filter), reinterpret_cast<const uint32_t*>(shader_frag_filter)});

    // --- 2. Create Pipeline Layouts ---
    std::array filter_layout_sets = { m_ubo_layout, m_texture_layout };
    m_filter_layout = m_device.createPipelineLayoutUnique({{}, filter_layout_sets});
    m_passthrough_layout = m_device.createPipelineLayoutUnique({{}, m_texture_layout});

    // --- 3. Define Shader Stages ---
    vk::PipelineShaderStageCreateInfo vert_stage_passthrough_ci({}, vk::ShaderStageFlagBits::eVertex, m_vert_passthrough.get(), "main");
    vk::PipelineShaderStageCreateInfo frag_stage_passthrough_ci({}, vk::ShaderStageFlagBits::eFragment, m_frag_passthrough.get(), "main");
    vk::PipelineShaderStageCreateInfo frag_stage_filter_ci({}, vk::ShaderStageFlagBits::eFragment, m_frag_shader_filter.get(), "main");

    // --- 4. Define Common Pipeline State (same as RmlRenderInterface) ---
    vk::VertexInputBindingDescription binding_desc(0, sizeof(Rml::Vertex), vk::VertexInputRate::eVertex);
    std::array attribute_descs = {
        vk::VertexInputAttributeDescription(0,0, vk::Format::eR32G32Sfloat, offsetof(Rml::Vertex, position)),
        vk::VertexInputAttributeDescription(1, 0, vk::Format::eR8G8B8A8Unorm, offsetof(Rml::Vertex, colour) ),
        vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Rml::Vertex, tex_coord))
    };
    vk::PipelineVertexInputStateCreateInfo vertex_input_ci({}, binding_desc, attribute_descs);
    vk::PipelineInputAssemblyStateCreateInfo input_assembly_ci({}, vk::PrimitiveTopology::eTriangleList);
    vk::PipelineViewportStateCreateInfo viewport_state_ci({}, 1, nullptr, 1, nullptr);
    vk::PipelineRasterizationStateCreateInfo rasterization_ci({}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise);
    vk::PipelineMultisampleStateCreateInfo multisample_ci({}, vk::SampleCountFlagBits::e1);
    std::array dynamic_states = { vk::DynamicState::eViewport, vk::DynamicState::eScissor, vk::DynamicState::eStencilReference };
    vk::PipelineDynamicStateCreateInfo dynamic_state_ci({}, dynamic_states);
    vk::PipelineRenderingCreateInfo rendering_ci({}, color_format, depth_format, depth_format);
    vk::PipelineColorBlendAttachmentState color_blend_attachment(true, vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    vk::PipelineColorBlendStateCreateInfo color_blend_ci({}, false, vk::LogicOp::eCopy, color_blend_attachment);
    vk::PipelineDepthStencilStateCreateInfo ds_no_stencil({}, false, false);
    vk::PipelineDepthStencilStateCreateInfo ds_stencil_clip({}, true, true, {}, {}, {}, {}, {vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eEqual, 0xFF, 0xFF, 1});
    ds_stencil_clip.back = ds_stencil_clip.front;

    // --- 5. Create Filter & Passthrough Pipelines ---
    vk::GraphicsPipelineCreateInfo pipeline_ci;
    pipeline_ci.pNext = &rendering_ci;
    pipeline_ci.pVertexInputState = &vertex_input_ci;
    pipeline_ci.pInputAssemblyState = &input_assembly_ci;
    pipeline_ci.pViewportState = &viewport_state_ci;
    pipeline_ci.pRasterizationState = &rasterization_ci;
    pipeline_ci.pMultisampleState = &multisample_ci;
    pipeline_ci.pDynamicState = &dynamic_state_ci;
    pipeline_ci.stageCount = 2;
    pipeline_ci.pStages = std::array{ vert_stage_passthrough_ci, frag_stage_filter_ci }.data();
    pipeline_ci.layout = m_filter_layout.get();
    pipeline_ci.pColorBlendState = &color_blend_ci;
    pipeline_ci.pDepthStencilState = &ds_no_stencil;
    m_filter_pipeline = m_device.createGraphicsPipelineUnique({}, pipeline_ci).value;
    pipeline_ci.pDepthStencilState = &ds_stencil_clip;
    m_filter_stencil_clip_pipeline = m_device.createGraphicsPipelineUnique({}, pipeline_ci).value;

    pipeline_ci.layout = m_passthrough_layout.get();
    pipeline_ci.pStages = std::array{ vert_stage_passthrough_ci, frag_stage_passthrough_ci }.data();
    pipeline_ci.pDepthStencilState = &ds_no_stencil;
    pipeline_ci.pColorBlendState = &color_blend_ci;
    m_passthrough_pipeline = m_device.createGraphicsPipelineUnique({}, pipeline_ci).value;
    
    vk::PipelineColorBlendAttachmentState no_blend_attachment;
    vk::PipelineColorBlendStateCreateInfo no_blend_ci({}, false, vk::LogicOp::eCopy, no_blend_attachment);
    pipeline_ci.pColorBlendState = &no_blend_ci;
    m_passthrough_replace_pipeline = m_device.createGraphicsPipelineUnique({}, pipeline_ci).value;
}

void RenderLayerManager::beginFrame(vk::CommandBuffer command_buffer, vk::Fence in_flight_fence, vk::Extent2D target_extent) {
    m_command_buffer = command_buffer;
    m_current_frame_fence = in_flight_fence;
    m_target_extent = target_extent;

    // Garbage collection for layers used in previous frames
    if(m_destruction_map_layers.contains(m_current_frame_fence)) {
        m_destruction_map_layers.erase(m_current_frame_fence);
    }
    
    // Check if post-process buffers need resizing
    if (m_postprocess_buffers[0] && (m_postprocess_buffers[0]->extent.width != m_target_extent.width || m_postprocess_buffers[0]->extent.height != m_target_extent.height)) {
        CreatePostProcessResources(m_target_extent);
    }

    RMLUI_ASSERT(m_layers.empty());
}

void RenderLayerManager::endFrame(vk::ImageView target_image_view) {
    RMLUI_ASSERT(m_layers.size() == 1);
    
    m_command_buffer.endRendering(); // End rendering to the base layer
    RenderLayer& base_layer = m_layers.back();

    // Transition base layer for shader reading
    vk::ImageMemoryBarrier to_shader_read(vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, base_layer.image, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    m_command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, to_shader_read);

    // Begin final render pass to the screen
    vk::RenderingAttachmentInfo color_attachment(target_image_view, vk::ImageLayout::eColorAttachmentOptimal,
        {}, {}, {}, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eStore);
    vk::RenderingInfo rendering_info({}, {{0, 0}, m_target_extent}, 1, 0, color_attachment);
    m_command_buffer.beginRendering(rendering_info);

    // Draw the final UI with the passthrough pipeline
    m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_passthrough_pipeline.get());
    m_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_passthrough_layout.get(), 0, base_layer.texture_descriptor_set, {});
    auto* quad_geom = reinterpret_cast<GeometryData*>(m_fullscreen_quad);
    m_command_buffer.bindVertexBuffers(0, quad_geom->getBuffer(), { quad_geom->getVertexOffset() });
    m_command_buffer.bindIndexBuffer(quad_geom->getBuffer(), quad_geom->getIndexOffset(), vk::IndexType::eUint32);
    m_command_buffer.drawIndexed(quad_geom->getNumIndices(), 1, 0, 0, 0);

    m_command_buffer.endRendering();

    // Schedule the base layer for destruction
    m_destruction_map_layers[m_current_frame_fence].push_back(std::move(m_layers.back()));
    m_layers.pop_back();
    RMLUI_ASSERT(m_layers.empty());
}

Rml::LayerHandle RenderLayerManager::PushLayer() {
    if (!m_layers.empty()) {
        m_command_buffer.endRendering();
    }
    
    constexpr vk::Format layer_format = vk::Format::eR8G8B8A8Unorm;
    
    m_layers.emplace_back(m_allocator, m_device, m_target_extent, layer_format, m_descriptor_pool, m_texture_layout, m_sampler);
    const Rml::LayerHandle handle = (Rml::LayerHandle)(m_layers.size() - 1);

    // Begin render pass for the new layer
    vk::RenderingAttachmentInfo color_attachment(
        m_layers.back().image_view.get(), vk::ImageLayout::eColorAttachmentOptimal,
        {}, {}, {}, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
        vk::ClearColorValue{ std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f} }
    );
    vk::RenderingInfo rendering_info({}, {{0, 0}, m_target_extent}, 1, 0, color_attachment);
    m_command_buffer.beginRendering(rendering_info);

    return handle;
}

void RenderLayerManager::PopLayer() {
    if (m_layers.empty()) return;
    
    m_command_buffer.endRendering();
    
    m_destruction_map_layers[m_current_frame_fence].push_back(std::move(m_layers.back()));
    m_layers.pop_back();
    
    // Resume rendering to the layer below
    if (!m_layers.empty()) {
        const vk::Extent2D layer_extent = m_layers.back().extent;
        vk::RenderingAttachmentInfo color_attachment(
            m_layers.back().image_view.get(), vk::ImageLayout::eColorAttachmentOptimal,
            {}, {}, {}, vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore
        );
        vk::RenderingInfo rendering_info({}, {{0, 0}, layer_extent}, 1, 0, color_attachment);
        m_command_buffer.beginRendering(rendering_info);
    }
}

void RenderLayerManager::CompositeLayers(Rml::LayerHandle source_handle, Rml::LayerHandle destination_handle, Rml::BlendMode blend_mode, Rml::Span<const Rml::CompiledFilterHandle> filters) {
    RenderLayer& source = m_layers.at(source_handle);
    RenderLayer& destination = m_layers.at(destination_handle);
    const vk::Extent2D extent = destination.extent;

    // --- Step 1: Blit source to the first post-process buffer ---
    m_postprocess_idx = 0;
    RenderLayer& postprocess_start = *m_postprocess_buffers[m_postprocess_idx];

    std::array<vk::ImageMemoryBarrier, 2> pre_blit_barriers = {
        vk::ImageMemoryBarrier(vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eTransferRead, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, source.image, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}),
        vk::ImageMemoryBarrier(vk::AccessFlagBits::eNone, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, postprocess_start.image, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
    };
    m_command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTransfer, {}, nullptr, nullptr, pre_blit_barriers);
    
    vk::ImageBlit blit_region{};
    blit_region.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    blit_region.srcOffsets[1] = vk::Offset3D{ (int32_t)extent.width, (int32_t)extent.height, 1 };
    blit_region.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    blit_region.dstOffsets[1] = vk::Offset3D{ (int32_t)extent.width, (int32_t)extent.height, 1 };
    m_command_buffer.blitImage(source.image, vk::ImageLayout::eTransferSrcOptimal, postprocess_start.image, vk::ImageLayout::eTransferDstOptimal, blit_region, vk::Filter::eNearest);

    vk::ImageMemoryBarrier post_blit_barrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, postprocess_start.image, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    m_command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, post_blit_barrier);

    // --- Step 2: Apply filters ---
    if (!filters.empty()) {
        RenderFilters(filters);
    }

    // --- Step 3: Composite result onto the destination layer ---
    RenderLayer& final_filtered_image = *m_postprocess_buffers[m_postprocess_idx];
    vk::RenderingAttachmentInfo color_attachment(destination.image_view.get(), vk::ImageLayout::eColorAttachmentOptimal, {}, {}, {}, vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore);
    vk::RenderingInfo rendering_info({}, {{0, 0}, extent}, 1, 0, color_attachment);
    m_command_buffer.beginRendering(rendering_info);

    const vk::Pipeline pipeline = (blend_mode == Rml::BlendMode::Replace) ? m_passthrough_replace_pipeline.get() : m_passthrough_pipeline.get();
    m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
    m_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_passthrough_layout.get(), 0, final_filtered_image.texture_descriptor_set, {});
    
    auto* quad_geom = reinterpret_cast<GeometryData*>(m_fullscreen_quad);
    m_command_buffer.bindVertexBuffers(0, quad_geom->getBuffer(), { quad_geom->getVertexOffset() });
    m_command_buffer.bindIndexBuffer(quad_geom->getBuffer(), quad_geom->getIndexOffset(), vk::IndexType::eUint32);
    m_command_buffer.drawIndexed(quad_geom->getNumIndices(), 1, 0, 0, 0);

    m_command_buffer.endRendering();
}

Rml::CompiledFilterHandle RenderLayerManager::CompileFilter(const Rml::String& name, const Rml::Dictionary& parameters) {
    auto compiled_filter = std::make_unique<CompiledFilter>();
    FilterData filter_data = {};
    if (name == "opacity")
    {
        compiled_filter->type = FilterType::Opacity;
        filter_data.filter_type = FILTER_TYPE_OPACITY;
        filter_data.scalar_value = Rml::Get(parameters, "value", 1.0f);
    }
    else if (name == "blur")
    {
        compiled_filter->type = FilterType::Blur;
        compiled_filter->sigma = Rml::Get(parameters, "sigma", 1.0f);
    }
    else if (name == "drop-shadow")
    {
        compiled_filter->type = FilterType::DropShadow;
        compiled_filter->sigma = Rml::Get(parameters, "sigma", 0.f);
        compiled_filter->drop_shadow_color = Rml::Get(parameters, "color", Rml::Colourb()).ToPremultiplied();
        compiled_filter->drop_shadow_offset = Rml::Get(parameters, "offset", Rml::Vector2f(0.f));
    }
    else if (name == "brightness")
    {
        compiled_filter->type = FilterType::ColorMatrix;
        filter_data.filter_type = FILTER_TYPE_COLOR_MATRIX;
        const float value = Rml::Get(parameters, "value", 1.0f);
        filter_data.color_matrix = Rml::Matrix4f::Diag(value, value, value, 1.f);
    }
    else if (name == "contrast")
    {
        compiled_filter->type = FilterType::ColorMatrix;
        filter_data.filter_type = FILTER_TYPE_COLOR_MATRIX;
        const float value = Rml::Get(parameters, "value", 1.0f);
        const float grayness = 0.5f - 0.5f * value;
        filter_data.color_matrix = Rml::Matrix4f::Diag(value, value, value, 1.f);
        filter_data.color_matrix.SetColumn(3, Rml::Vector4f(grayness, grayness, grayness, 0.f));
    }
    else if (name == "invert")
    {
        compiled_filter->type = FilterType::ColorMatrix;
        filter_data.filter_type = FILTER_TYPE_COLOR_MATRIX;
        const float value = Rml::Math::Clamp(Rml::Get(parameters, "value", 1.0f), 0.f, 1.f);
        const float inverted = 1.f - 2.f * value;
        filter_data.color_matrix = Rml::Matrix4f::Diag(inverted, inverted, inverted, 1.f);
        filter_data.color_matrix.SetColumn(3, Rml::Vector4f(value, value, value, 0.f));
    }
    else if (name == "grayscale")
    {
        compiled_filter->type = FilterType::ColorMatrix;
        filter_data.filter_type = FILTER_TYPE_COLOR_MATRIX;
        const float value = Rml::Get(parameters, "value", 1.0f);
        const float rev_value = 1.f - value;
        const Rml::Vector3f gray = value * Rml::Vector3f(0.2126f, 0.7152f, 0.0722f);
        filter_data.color_matrix = Rml::Matrix4f::FromRows(
            {gray.x + rev_value, gray.y,              gray.z,              0.f},
            {gray.x,              gray.y + rev_value, gray.z,              0.f},
            {gray.x,              gray.y,              gray.z + rev_value, 0.f},
            {0.f,                 0.f,                 0.f,                 1.f}
        );
    }
    else
    {
        Rml::Log::Message(Rml::Log::LT_WARNING, "Unsupported filter type '%s'.", name.c_str());
        return {};
    }

    auto ubo = UniformBuffer::Create<FilterData>(filter_data, m_allocator, m_device, m_descriptor_pool, m_ubo_layout);
    compiled_filter->ubo = std::move(ubo);
    return reinterpret_cast<Rml::CompiledFilterHandle>(compiled_filter.release());
}

void RenderLayerManager::ReleaseFilter(Rml::CompiledFilterHandle filter_handle) {
    if (filter_handle)
        delete reinterpret_cast<CompiledFilter*>(filter_handle);
}

void RenderLayerManager::RenderFilters(Rml::Span<const Rml::CompiledFilterHandle> filters) {
    for (const auto& filter_handle : filters)
    {
        const auto* filter = reinterpret_cast<const CompiledFilter*>(filter_handle);
        if (!filter) continue;

        switch (filter->type)
        {
            case FilterType::ColorMatrix:
            case FilterType::Opacity:
            {
                RenderSingleFilterPass(filter);
                break;
            }

            case FilterType::Blur:
            {
                FilterData blur_data = {};
                CalculateBlurWeights(filter->sigma, blur_data);

                blur_data.filter_type = FILTER_TYPE_BLUR_VERTICAL;
                filter->ubo->Update(blur_data);
                RenderSingleFilterPass(filter);

                blur_data.filter_type = FILTER_TYPE_BLUR_HORIZONTAL;
                filter->ubo->Update(blur_data);
                RenderSingleFilterPass(filter);
                break;
            }

            case FilterType::DropShadow:
            {
                RenderLayer& source_image = *m_postprocess_buffers[m_postprocess_idx];

                FilterData shadow_data = {};
                shadow_data.filter_type = FILTER_TYPE_DROP_SHADOW_ALPHA;
                shadow_data.drop_shadow_offset = filter->drop_shadow_offset;
                Rml::Colourf converted_color;
                for (int i = 0; i < 4; ++i) converted_color[i] = static_cast<float>(filter->drop_shadow_color[i]) / 255.0f;
                shadow_data.drop_shadow_color = converted_color;
                
                filter->ubo->Update(shadow_data);
                RenderSingleFilterPass(filter);

                if (filter->sigma > 0.1f) {
                    FilterData blur_data = {};
                    CalculateBlurWeights(filter->sigma, blur_data);
                    blur_data.filter_type = FILTER_TYPE_BLUR_VERTICAL;
                    filter->ubo->Update(blur_data);
                    RenderSingleFilterPass(filter);
                    blur_data.filter_type = FILTER_TYPE_BLUR_HORIZONTAL;
                    filter->ubo->Update(blur_data);
                    RenderSingleFilterPass(filter);
                }

                RenderLayer& shadow_image = *m_postprocess_buffers[m_postprocess_idx];
                
                vk::RenderingAttachmentInfo color_attachment(shadow_image.image_view.get(), vk::ImageLayout::eColorAttachmentOptimal, {}, {}, {}, vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore);
                vk::RenderingInfo rendering_info({}, {{0, 0}, m_target_extent}, 1, 0, color_attachment);
                m_command_buffer.beginRendering(rendering_info);

                m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_passthrough_pipeline.get());
                m_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_passthrough_layout.get(), 0, source_image.texture_descriptor_set, {});
                
                auto* quad_geom = reinterpret_cast<GeometryData*>(m_fullscreen_quad);
                m_command_buffer.bindVertexBuffers(0, quad_geom->getBuffer(), { quad_geom->getVertexOffset() });
                m_command_buffer.bindIndexBuffer(quad_geom->getBuffer(), quad_geom->getIndexOffset(), vk::IndexType::eUint32);
                m_command_buffer.drawIndexed(quad_geom->getNumIndices(), 1, 0, 0, 0);

                m_command_buffer.endRendering();
                break;
            }
            default: break;
        }
    }
}

void RenderLayerManager::RenderSingleFilterPass(const CompiledFilter* filter) {
    RenderLayer& read_buffer = *m_postprocess_buffers[m_postprocess_idx];
    RenderLayer& write_buffer = *m_postprocess_buffers[1 - m_postprocess_idx];

    vk::RenderingAttachmentInfo color_attachment(write_buffer.image_view.get(), vk::ImageLayout::eColorAttachmentOptimal, {}, {}, {}, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::ClearValue(vk::ClearColorValue(std::array{0.f, 0.f, 0.f, 0.f})));
    vk::RenderingInfo rendering_info({}, {{0, 0}, m_target_extent}, 1, 0, color_attachment);
    m_command_buffer.beginRendering(rendering_info);

    const vk::Pipeline pipeline = m_clip_mask_enabled ? m_filter_stencil_clip_pipeline.get() : m_filter_pipeline.get();
    m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

    const std::array sets = {
        filter->ubo->getDescriptorSet(),
        read_buffer.texture_descriptor_set
    };
    m_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_filter_layout.get(), 0, sets, {});
    
    auto* quad_geom = reinterpret_cast<GeometryData*>(m_fullscreen_quad);
    m_command_buffer.bindVertexBuffers(0, quad_geom->getBuffer(), { quad_geom->getVertexOffset() });
    m_command_buffer.bindIndexBuffer(quad_geom->getBuffer(), quad_geom->getIndexOffset(), vk::IndexType::eUint32);
    m_command_buffer.drawIndexed(quad_geom->getNumIndices(), 1, 0, 0, 0);

    m_command_buffer.endRendering();

    const vk::ImageMemoryBarrier barrier(vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, write_buffer.image, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    m_command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

    m_postprocess_idx = 1 - m_postprocess_idx;
}

void RenderLayerManager::CalculateBlurWeights(float sigma, FilterData& out_data) {
    constexpr int num_weights = sizeof(out_data.blur_weights) / sizeof(float);
    float normalization = 0.0f;
    for (int i = 0; i < num_weights; i++)
    {
        if (sigma < 0.1f) {
            out_data.blur_weights[i] = (i == 0) ? 1.0f : 0.0f;
        } else {
            out_data.blur_weights[i] = expf(-float(i * i) / (2.0f * sigma * sigma));
        }
        normalization += (i == 0 ? 1.0f : 2.0f) * out_data.blur_weights[i];
    }
    for (int i = 0; i < num_weights; i++) {
        out_data.blur_weights[i] /= normalization;
    }
}