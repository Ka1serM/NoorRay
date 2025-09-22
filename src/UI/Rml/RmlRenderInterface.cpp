#include "RmlRenderInterface.h"
#include "RenderLayerManager.h"
#include "GeometryData.h"       // Include the new header
#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/FileInterface.h>
#include <RmlUi/Core/Log.h>
#include <RmlUi/Core/Math.h>
#include <array>
#include <cmath>

#include "Log.h"
#include "TextureData.h"
#include "UniformBuffer.h"
#include "RmlUi/Core/DecorationTypes.h"
#include "RmlUi/Core/Dictionary.h"

RmlRenderInterface::RmlRenderInterface(
    const vk::Device device,
    const vk::Queue graphics_queue,
    const VmaAllocator allocator,
    const vk::CommandPool command_pool,
    const vk::DescriptorPool descriptor_pool,
    const vk::Format colorFormat,
    const vk::Format depthFormat,
    const vk::Extent2D initialExtent
) : m_device(device),
    m_graphics_queue(graphics_queue),
    m_allocator(allocator),
    m_utility_command_pool(command_pool),
    m_descriptor_pool(descriptor_pool)
{
    CreateDescriptors();
    CreatePipelines(colorFormat, depthFormat);

    Rml::Vertex vertices[] = {
        {{-1.0f, -1.0f}, {}, {0.0f, 1.0f}},
        {{1.0f, -1.0f}, {}, {1.0f, 1.0f}},
        {{1.0f, 1.0f}, {}, {1.0f, 0.0f}},
        {{-1.0f, 1.0f}, {}, {0.0f, 0.0f}}
    };
    int indices[] = {0, 1, 2, 0, 2, 3};
    m_fullscreen_quad = CompileGeometry({vertices, 4}, {indices, 6});

    m_render_layer_manager = std::make_unique<RenderLayerManager>(
        m_allocator,
        m_device,
        m_descriptor_pool,
        m_texture_descriptor_set_layout.get(),
        m_ubo_descriptor_set_layout.get(),
        m_linear_sampler.get(),
        m_fullscreen_quad,
        colorFormat,
        depthFormat,
        initialExtent
    );
}

void RmlRenderInterface::CreateDescriptors()
{
    const vk::SamplerCreateInfo sampler_ci({}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat);
    m_linear_sampler = m_device.createSamplerUnique(sampler_ci);

    vk::DescriptorSetLayoutBinding textures_binding(0, vk::DescriptorType::eCombinedImageSampler, kMaxBindlessTextures, vk::ShaderStageFlagBits::eFragment);
    vk::DescriptorBindingFlags binding_flags = vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind;
    const vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_ci(binding_flags);
    vk::DescriptorSetLayoutCreateInfo textures_layout_ci(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool, textures_binding);
    textures_layout_ci.pNext = &flags_ci;
    m_textures_descriptor_set_layout = m_device.createDescriptorSetLayoutUnique(textures_layout_ci);

    const vk::DescriptorSetAllocateInfo alloc_info(m_descriptor_pool, m_textures_descriptor_set_layout.get());
    m_textures_descriptor_set = std::move(m_device.allocateDescriptorSetsUnique(alloc_info)[0]);

    m_free_texture_indices.resize(kMaxBindlessTextures);
    for (uint32_t i = 0; i < kMaxBindlessTextures; ++i)
        m_free_texture_indices[i] = kMaxBindlessTextures - 1 - i;

    vk::DescriptorSetLayoutBinding ubo_binding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment);
    const vk::DescriptorSetLayoutCreateInfo ubo_layout_ci({}, ubo_binding);
    m_ubo_descriptor_set_layout = m_device.createDescriptorSetLayoutUnique(ubo_layout_ci);

    vk::DescriptorSetLayoutBinding texture_binding(0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment);
    const vk::DescriptorSetLayoutCreateInfo texture_layout_ci({}, texture_binding);
    m_texture_descriptor_set_layout = m_device.createDescriptorSetLayoutUnique(texture_layout_ci);
}

RmlRenderInterface::~RmlRenderInterface()
{
    if (m_device)
    {
        try
        {
            m_device.waitIdle();

            if (m_fullscreen_quad)
            {
                delete reinterpret_cast<GeometryData*>(m_fullscreen_quad);
                m_fullscreen_quad = {};
            }

            m_destruction_map_geometry.clear();
            m_destruction_map_textures.clear();
            m_registered_textures.clear();
        }
        catch (const vk::SystemError& err)
        {
            Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan device waitIdle failed in destructor: %s", err.what());
        }
    }
}

void RmlRenderInterface::beginFrame(const vk::CommandBuffer command_buffer, const vk::Image target_image, const vk::ImageView target_image_view, const vk::ImageView depthImageView, vk::Extent2D target_extent, vk::Fence in_flight_fence)
{
    m_current_frame_fence = in_flight_fence;
    m_command_buffer = command_buffer;
    m_target_extent = target_extent;
    m_target_image_view = target_image_view;
    m_target_image = target_image;

    if (m_destruction_map_textures.contains(m_current_frame_fence))
    {
        m_destruction_map_textures.erase(m_current_frame_fence);
    }
    if (m_destruction_map_geometry.contains(m_current_frame_fence))
    {
        m_destruction_map_geometry.erase(m_current_frame_fence);
    }

    m_clip_mask_enabled = false;
    m_transform_enabled = false;
    m_transform_matrix = Rml::Matrix4f::Identity();

    const float w = static_cast<float>(target_extent.width);
    const float h = static_cast<float>(target_extent.height);
    m_projection_matrix = Rml::Matrix4f::ProjectOrtho(0.0f, w, h, 0.0f, -10000.0f, 10000.0f);
    m_projection_matrix[1][1] *= -1.0f;

#if RMLUI_VULKAN_ENABLE_LAYERS
    m_render_layer_manager->beginFrame(command_buffer, in_flight_fence, target_extent);
    PushLayer();
#else
    vk::RenderingAttachmentInfo color_attachment(
        m_target_image_view, vk::ImageLayout::eColorAttachmentOptimal,
        {}, {}, {},
        vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
        vk::ClearColorValue{ std::array{0.0f, 0.0f, 0.0f, 0.0f} }
    );
    const vk::RenderingInfo rendering_info({}, {{0, 0}, m_target_extent}, 1, 0, color_attachment);
    m_command_buffer.beginRendering(rendering_info);
#endif
}

void RmlRenderInterface::endFrame()
{
#if RMLUI_VULKAN_ENABLE_LAYERS
    PopLayer();
    m_render_layer_manager->endFrame(m_target_image_view);
#else
    m_command_buffer.endRendering();
#endif
}

Rml::CompiledGeometryHandle RmlRenderInterface::CompileGeometry(const Rml::Span<const Rml::Vertex> vertices, const Rml::Span<const int> indices)
{
    return reinterpret_cast<Rml::CompiledGeometryHandle>(new GeometryData(m_allocator, vertices, indices));
}

void RmlRenderInterface::RenderGeometry(Rml::CompiledGeometryHandle geometry_handle, Rml::Vector2f translation, Rml::TextureHandle texture)
{
    auto* geometry = reinterpret_cast<GeometryData*>(geometry_handle);
    if (!geometry)
        return;

    PushConstants pc{};
    pc.transform = m_projection_matrix * m_transform_matrix;
    pc.translate = translation;
    if (texture)
        pc.texture_id = reinterpret_cast<TextureData*>(texture)->getBindlessIndex();

    const vk::Pipeline pipeline = m_clip_mask_enabled ? m_pipeline_stencil_clip.get() : m_pipeline_main.get();
    m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

    m_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipeline_textures_layout.get(), 0, m_textures_descriptor_set.get(), {});

    m_command_buffer.pushConstants<PushConstants>(m_pipeline_textures_layout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);

    m_command_buffer.bindVertexBuffers(0, geometry->getBuffer(), {geometry->getVertexOffset()});
    m_command_buffer.bindIndexBuffer(geometry->getBuffer(), geometry->getIndexOffset(), vk::IndexType::eUint32);
    m_command_buffer.drawIndexed(geometry->getNumIndices(), 1, 0, 0, 0);
}

void RmlRenderInterface::ReleaseGeometry(const Rml::CompiledGeometryHandle geometry_handle)
{
    if (geometry_handle && m_current_frame_fence)
    {
        // Safely transfer ownership of the raw pointer to a unique_ptr in the destruction map.
        m_destruction_map_geometry[m_current_frame_fence].emplace_back(reinterpret_cast<GeometryData*>(geometry_handle));
    }
}

Rml::TextureHandle RmlRenderInterface::GenerateTexture(const Rml::Span<const Rml::byte> source_data, const Rml::Vector2i source_dimensions)
{
    if (m_free_texture_indices.empty())
    {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create texture: no free bindless indices available.");
        return {};
    }

    auto texture_ptr = new TextureData(m_allocator, m_device, m_graphics_queue, m_utility_command_pool,
                                       source_data, source_dimensions, m_linear_sampler.get());

    const uint32_t id = m_free_texture_indices.back();
    m_free_texture_indices.pop_back();
    texture_ptr->setBindlessIndex(id);

    vk::DescriptorImageInfo image_info(texture_ptr->getSampler(), texture_ptr->getImageView(), vk::ImageLayout::eShaderReadOnlyOptimal);
    vk::WriteDescriptorSet write(m_textures_descriptor_set.get(), 0, id, vk::DescriptorType::eCombinedImageSampler, image_info);
    m_device.updateDescriptorSets(write, {});

    return reinterpret_cast<Rml::TextureHandle>(texture_ptr);
}

void RmlRenderInterface::registerVulkanTexture(const Rml::String& name, const vk::ImageView image_view, const Rml::Vector2i dimensions)
{
    if (m_registered_textures.contains(name))
    {
        Rml::Log::Message(Rml::Log::LT_WARNING, "A Vulkan texture with the name '%s' is already registered. Ignoring request.", name.c_str());
        return;
    }

    auto texture_ptr = new TextureData(m_allocator, m_device, image_view);

    if (m_free_texture_indices.empty()) { return; }
    uint32_t id = m_free_texture_indices.back();
    m_free_texture_indices.pop_back();
    texture_ptr->setBindlessIndex(id);

    vk::DescriptorImageInfo image_info(texture_ptr->getSampler(), texture_ptr->getImageView(), vk::ImageLayout::eShaderReadOnlyOptimal);
    vk::WriteDescriptorSet write(m_textures_descriptor_set.get(), 0, id, vk::DescriptorType::eCombinedImageSampler, image_info);
    m_device.updateDescriptorSets(write, {});

    Rml::TextureHandle handle = reinterpret_cast<Rml::TextureHandle>(texture_ptr);
    m_registered_textures[name] = {handle, dimensions};
}

void RmlRenderInterface::ReleaseTexture(const Rml::TextureHandle texture_handle)
{
    if (texture_handle && m_current_frame_fence)
    {
        auto* texture_ptr = reinterpret_cast<TextureData*>(texture_handle);

        if (texture_ptr->getBindlessIndex() != static_cast<uint32_t>(-1))
        {
            m_free_texture_indices.push_back(texture_ptr->getBindlessIndex());
        }

        m_destruction_map_textures[m_current_frame_fence].emplace_back(texture_ptr);
    }
}

void RmlRenderInterface::unregisterVulkanTexture(const Rml::String& name)
{
    auto it = m_registered_textures.find(name);
    if (it != m_registered_textures.end())
    {
        Rml::TextureHandle handle = it->second.first;
        ReleaseTexture(handle);
        m_registered_textures.erase(it);
    }
}

Rml::TextureHandle RmlRenderInterface::LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source)
{
    LOG_INFO("Loading Texture" + source);

    if (source.rfind("vulkan://", 0) == 0)
    {
        const Rml::String texture_name = source.substr(9);
        const auto it = m_registered_textures.find(texture_name);

        if (it != m_registered_textures.end())
        {
            texture_dimensions = it->second.second;
            return it->second.first;
        }
    }
    Rml::FileInterface* file_interface = Rml::GetFileInterface();
    const Rml::FileHandle file_handle = file_interface->Open(source);
    if (!file_handle) return {};

    file_interface->Seek(file_handle, 0, SEEK_END);
    const size_t buffer_size = file_interface->Tell(file_handle);
    file_interface->Seek(file_handle, 0, SEEK_SET);

    if (buffer_size <= sizeof(TGAHeader))
    {
        file_interface->Close(file_handle);
        return {};
    }

    const auto buffer = std::unique_ptr<Rml::byte[]>(new Rml::byte[buffer_size]);
    file_interface->Read(buffer.get(), buffer_size, file_handle);
    file_interface->Close(file_handle);

    TGAHeader header{};
    memcpy(&header, buffer.get(), sizeof(TGAHeader));

    if (header.dataType != 2 || (header.bitsPerPixel != 24 && header.bitsPerPixel != 32))
    {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Unsupported TGA: %s. Only 24/32bit uncompressed supported.", source.c_str());
        return {};
    }

    texture_dimensions = {header.width, header.height};
    const int color_mode = header.bitsPerPixel / 8;
    const size_t image_size = static_cast<size_t>(header.width) * header.height * 4;
    auto image_dest = std::unique_ptr<Rml::byte[]>(new Rml::byte[image_size]);
    const Rml::byte* image_src = buffer.get() + sizeof(TGAHeader);

    for (int y = 0; y < header.height; ++y)
    {
        for (int x = 0; x < header.width; ++x)
        {
            const int read_idx = (y * header.width + x) * color_mode;
            const int write_idx = ((!(header.imageDescriptor & 32) ? (header.height - 1 - y) : y) * header.width + x) * 4;
            image_dest[write_idx] = image_src[read_idx + 2];
            image_dest[write_idx + 1] = image_src[read_idx + 1];
            image_dest[write_idx + 2] = image_src[read_idx];
            if (color_mode == 4)
            {
                const Rml::byte alpha = image_src[read_idx + 3];
                image_dest[write_idx] = Rml::byte((image_dest[write_idx] * alpha) / 255);
                image_dest[write_idx + 1] = Rml::byte((image_dest[write_idx + 1] * alpha) / 255);
                image_dest[write_idx + 2] = Rml::byte((image_dest[write_idx + 2] * alpha) / 255);
                image_dest[write_idx + 3] = alpha;
            }
            else
                image_dest[write_idx + 3] = 255;
        }
    }

    return GenerateTexture({image_dest.get(), image_size}, texture_dimensions);
}

void RmlRenderInterface::SetScissorRegion(const Rml::Rectanglei region)
{
    if (!m_scissor_enabled) return;

    if (m_transform_enabled)
    {
        EnableClipMask(true);

        Rml::Vertex vertices[4];
        vertices[0].position = {static_cast<float>(region.Left()), static_cast<float>(region.Top())};
        vertices[1].position = {static_cast<float>(region.Right()), static_cast<float>(region.Top())};
        vertices[2].position = {static_cast<float>(region.Right()), static_cast<float>(region.Bottom())};
        vertices[3].position = {static_cast<float>(region.Left()), static_cast<float>(region.Bottom())};
        int indices[] = {0, 1, 2, 0, 2, 3};

        if (const auto handle = CompileGeometry({vertices, 4}, {indices, 6}))
        {
            RenderToClipMask(Rml::ClipMaskOperation::Set, handle, {});
            ReleaseGeometry(handle);
        }
    }
    else
    {
        EnableClipMask(false);

        vk::Rect2D scissor;
        scissor.offset.x = Rml::Math::Max(0, region.Left());
        scissor.offset.y = Rml::Math::Max(0, region.Top());
        scissor.extent.width = region.Width();
        scissor.extent.height = region.Height();
        m_command_buffer.setScissor(0, scissor);
    }
}

void RmlRenderInterface::EnableScissorRegion(const bool enable)
{
    m_scissor_enabled = enable;
    if (!enable)
    {
        EnableClipMask(false);
        m_command_buffer.setScissor(0, m_current_scissor);
    }
}

void RmlRenderInterface::SetTransform(const Rml::Matrix4f* transform)
{
    m_transform_enabled = (transform != nullptr);
    m_transform_matrix = transform ? *transform : Rml::Matrix4f::Identity();
}

void RmlRenderInterface::CreatePipelines(vk::Format colorFormat, vk::Format depthFormat) {
    vk::PushConstantRange push_constant_range(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstants));

    m_pipeline_textures_layout = m_device.createPipelineLayoutUnique({{}, m_textures_descriptor_set_layout.get(), push_constant_range});
    
    // **FIX:** The texture layout (for set 0) must come before the UBO layout (for set 1).
    std::array ubo_textures_layout_sets = { m_textures_descriptor_set_layout.get(), m_ubo_descriptor_set_layout.get() };
    m_pipeline_ubo_textures_layout = m_device.createPipelineLayoutUnique({{}, ubo_textures_layout_sets, push_constant_range});

    vk::UniqueShaderModule vert_shader = m_device.createShaderModuleUnique({{}, sizeof(shader_vert), reinterpret_cast<const uint32_t*>(shader_vert)});
    vk::UniqueShaderModule frag_shader = m_device.createShaderModuleUnique({{}, sizeof(shader_frag), reinterpret_cast<const uint32_t*>(shader_frag)});
    vk::UniqueShaderModule frag_shader_gradient = m_device.createShaderModuleUnique({{}, sizeof(shader_frag_gradient), reinterpret_cast<const uint32_t*>(shader_frag_gradient)});
    
    vk::PipelineShaderStageCreateInfo vert_stage_ci({}, vk::ShaderStageFlagBits::eVertex, vert_shader.get(), "main");
    vk::PipelineShaderStageCreateInfo frag_stage_main_ci({}, vk::ShaderStageFlagBits::eFragment, frag_shader.get(), "main");
    vk::PipelineShaderStageCreateInfo frag_stage_gradient_ci({}, vk::ShaderStageFlagBits::eFragment, frag_shader_gradient.get(), "main");

    vk::VertexInputBindingDescription binding_desc(0, sizeof(Rml::Vertex), vk::VertexInputRate::eVertex);
    std::array attribute_descs = {
        vk::VertexInputAttributeDescription(0,0, vk::Format::eR32G32Sfloat, offsetof(Rml::Vertex, position)),
        vk::VertexInputAttributeDescription(1, 0, vk::Format::eR8G8B8A8Unorm, offsetof(Rml::Vertex, colour)),
        vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Rml::Vertex, tex_coord))
    };
    vk::PipelineVertexInputStateCreateInfo vertex_input_ci({}, binding_desc, attribute_descs);

    vk::PipelineInputAssemblyStateCreateInfo input_assembly_ci({}, vk::PrimitiveTopology::eTriangleList);
    vk::PipelineViewportStateCreateInfo viewport_state_ci({}, 1, nullptr, 1, nullptr);
    vk::PipelineRasterizationStateCreateInfo rasterization_ci({}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise);
    vk::PipelineMultisampleStateCreateInfo multisample_ci({}, vk::SampleCountFlagBits::e1);
    std::array dynamic_states = { vk::DynamicState::eViewport, vk::DynamicState::eScissor, vk::DynamicState::eStencilReference };
    vk::PipelineDynamicStateCreateInfo dynamic_state_ci({}, dynamic_states);
    vk::PipelineRenderingCreateInfo rendering_ci({}, colorFormat, depthFormat, depthFormat);
    
    vk::PipelineColorBlendAttachmentState color_blend_attachment(true, vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    vk::PipelineColorBlendStateCreateInfo color_blend_ci({}, false, vk::LogicOp::eCopy, color_blend_attachment);
    vk::PipelineColorBlendAttachmentState no_color_write_attachment; no_color_write_attachment.colorWriteMask = {};
    vk::PipelineColorBlendStateCreateInfo no_color_blend_ci({}, false, vk::LogicOp::eCopy, no_color_write_attachment);

    vk::PipelineDepthStencilStateCreateInfo ds_no_stencil({}, false, false);
    vk::PipelineDepthStencilStateCreateInfo ds_stencil_gen({}, true, true, {}, {}, {}, {}, {vk::StencilOp::eKeep, vk::StencilOp::eReplace, vk::StencilOp::eKeep, vk::CompareOp::eAlways, 0xFF, 0xFF, 1});
    ds_stencil_gen.back = ds_stencil_gen.front;
    vk::PipelineDepthStencilStateCreateInfo ds_stencil_clip({}, true, true, {}, {}, {}, {}, {vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eEqual, 0xFF, 0xFF, 1});
    ds_stencil_clip.back = ds_stencil_clip.front;
    vk::PipelineDepthStencilStateCreateInfo ds_stencil_incr({}, true, true, {}, {}, {}, {}, {vk::StencilOp::eKeep, vk::StencilOp::eIncrementAndClamp, vk::StencilOp::eKeep, vk::CompareOp::eEqual, 0xFF, 0xFF, 0});
    ds_stencil_incr.back = ds_stencil_incr.front;
    vk::PipelineDepthStencilStateCreateInfo ds_stencil_zero({}, true, true, {}, {}, {}, {}, {vk::StencilOp::eKeep, vk::StencilOp::eZero, vk::StencilOp::eKeep, vk::CompareOp::eAlways, 0xFF, 0xFF, 0});
    ds_stencil_zero.back = ds_stencil_zero.front;

    vk::GraphicsPipelineCreateInfo pipeline_ci;
    pipeline_ci.pNext = &rendering_ci;
    pipeline_ci.pVertexInputState = &vertex_input_ci;
    pipeline_ci.pInputAssemblyState = &input_assembly_ci;
    pipeline_ci.pViewportState = &viewport_state_ci;
    pipeline_ci.pRasterizationState = &rasterization_ci;
    pipeline_ci.pMultisampleState = &multisample_ci;
    pipeline_ci.pDynamicState = &dynamic_state_ci;
    pipeline_ci.stageCount = 2;

    // **FIX:** Complete pipeline creation
    std::array<vk::GraphicsPipelineCreateInfo, 7> pipeline_create_infos;

    // Main UI Pipelines
    pipeline_ci.layout = m_pipeline_textures_layout.get();
    pipeline_ci.pStages = std::array{ vert_stage_ci, frag_stage_main_ci }.data();
    pipeline_ci.pColorBlendState = &color_blend_ci;
    pipeline_ci.pDepthStencilState = &ds_no_stencil;
    pipeline_create_infos[0] = pipeline_ci;
    pipeline_ci.pColorBlendState = &no_color_blend_ci;
    pipeline_ci.pDepthStencilState = &ds_stencil_gen;
    pipeline_create_infos[1] = pipeline_ci;
    pipeline_ci.pDepthStencilState = &ds_stencil_incr;
    pipeline_create_infos[2] = pipeline_ci;
    pipeline_ci.pDepthStencilState = &ds_stencil_zero;
    pipeline_create_infos[3] = pipeline_ci;
    pipeline_ci.pColorBlendState = &color_blend_ci;
    pipeline_ci.pDepthStencilState = &ds_stencil_clip;
    pipeline_create_infos[4] = pipeline_ci;

    // Gradient Pipelines
    pipeline_ci.layout = m_pipeline_ubo_textures_layout.get();
    pipeline_ci.pStages = std::array{ vert_stage_ci, frag_stage_gradient_ci }.data();
    pipeline_ci.pColorBlendState = &color_blend_ci;
    pipeline_ci.pDepthStencilState = &ds_no_stencil;
    pipeline_create_infos[5] = pipeline_ci;
    pipeline_ci.pDepthStencilState = &ds_stencil_clip;
    pipeline_create_infos[6] = pipeline_ci;

    auto result = m_device.createGraphicsPipelinesUnique({}, pipeline_create_infos);
    if (result.result != vk::Result::eSuccess) {
         Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create UI pipelines: %s", vk::to_string(result.result).c_str());
    } else {
        m_pipeline_main = std::move(result.value[0]);
        m_pipeline_stencil_gen = std::move(result.value[1]);
        m_pipeline_stencil_incr = std::move(result.value[2]);
        m_pipeline_stencil_zero = std::move(result.value[3]);
        m_pipeline_stencil_clip = std::move(result.value[4]);
        m_pipeline_gradient = std::move(result.value[5]);
        m_pipeline_gradient_stencil_clip = std::move(result.value[6]);
    }
}

void RmlRenderInterface::EnableClipMask(const bool enable)
{
    m_clip_mask_enabled = enable;
    m_render_layer_manager->SetClipMaskEnabled(enable);
    if (!enable)
        m_stencil_level = 0;
}

void RmlRenderInterface::RenderToClipMask(Rml::ClipMaskOperation operation, const Rml::CompiledGeometryHandle handle, const Rml::Vector2f translation)
{
    const auto* geometry = reinterpret_cast<GeometryData*>(handle);
    if (!geometry) return;

    m_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipeline_textures_layout.get(), 0, m_textures_descriptor_set.get(), {});

    if (operation == Rml::ClipMaskOperation::SetInverse)
    {
        m_stencil_level = 1;
        m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline_stencil_gen.get());
        m_command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, m_stencil_level);

        const PushConstants quad_pc = {Rml::Matrix4f::Identity(), {0, 0}, -1};
        const auto* quad_geo = reinterpret_cast<GeometryData*>(m_fullscreen_quad);
        m_command_buffer.pushConstants<PushConstants>(m_pipeline_textures_layout.get(), vk::ShaderStageFlagBits::eVertex, 0, quad_pc);
        m_command_buffer.bindVertexBuffers(0, quad_geo->getBuffer(), {quad_geo->getVertexOffset()});
        m_command_buffer.bindIndexBuffer(quad_geo->getBuffer(), quad_geo->getIndexOffset(), vk::IndexType::eUint32);
        m_command_buffer.drawIndexed(quad_geo->getNumIndices(), 1, 0, 0, 0);

        m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline_stencil_zero.get());
        m_command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, 0);
    }
    else
    {
        switch (operation)
        {
        case Rml::ClipMaskOperation::Set:
            m_stencil_level = 1;
            m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline_stencil_gen.get());
            m_command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, m_stencil_level);
            break;

        case Rml::ClipMaskOperation::Intersect:
            m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline_stencil_incr.get());
            m_command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, m_stencil_level);
            m_stencil_level++;
            break;
        default: ;
        }
    }

    PushConstants constants = {};
    constants.transform = m_projection_matrix * m_transform_matrix,
        constants.translate = translation,
        m_command_buffer.pushConstants<PushConstants>(m_pipeline_textures_layout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, constants);
    m_command_buffer.bindVertexBuffers(0, geometry->getBuffer(), {geometry->getVertexOffset()});
    m_command_buffer.bindIndexBuffer(geometry->getBuffer(), geometry->getIndexOffset(), vk::IndexType::eUint32);
    m_command_buffer.drawIndexed(geometry->getNumIndices(), 1, 0, 0, 0);
}

Rml::CompiledShaderHandle RmlRenderInterface::CompileShader(const Rml::String& name, const Rml::Dictionary& parameters)
{
    GradientData gradient_data{};

    if (name == "linear-gradient" || name == "repeating-linear-gradient")
    {
        gradient_data.gradient_function = (name == "repeating-linear-gradient" ? 3 : 0);
        Rml::Vector2f p0 = Rml::Get(parameters, "p0", Rml::Vector2f(0.f));
        Rml::Vector2f p1 = Rml::Get(parameters, "p1", p0 + Rml::Vector2f(1.f, 0.f));
        gradient_data.p = p0;
        gradient_data.v = p1 - p0;
        if (gradient_data.v.SquaredMagnitude() < 1e-6f)
            gradient_data.v = {1.f, 0.f};
    }
    else if (name == "radial-gradient" || name == "repeating-radial-gradient")
    {
        gradient_data.gradient_function = (name == "repeating-radial-gradient" ? 4 : 1);
        gradient_data.p = Rml::Get(parameters, "center", Rml::Vector2f(0.f));
        Rml::Vector2f radius = Rml::Get(parameters, "radius", Rml::Vector2f(1.f));
        gradient_data.v = {1.f / (radius.x * radius.x), 1.f / (radius.y * radius.y)};
    }
    else if (name == "conic-gradient" || name == "repeating-conic-gradient")
    {
        gradient_data.gradient_function = (name == "repeating-conic-gradient" ? 5 : 2);
        gradient_data.p = Rml::Get(parameters, "center", Rml::Vector2f(0.f));
        float angle_rad = Rml::Get(parameters, "angle", 0.0f);
        gradient_data.v = {Rml::Math::Cos(angle_rad), Rml::Math::Sin(angle_rad)};
    }
    else
    {
        Rml::Log::Message(Rml::Log::LT_WARNING, "Unsupported shader type '%s'.", name.c_str());
        return {};
    }

    auto it = parameters.find("color_stop_list");
    RMLUI_ASSERT(it != parameters.end());
    const auto& color_stop_list = it->second.GetReference<Rml::ColorStopList>();
    const int num_stops = Rml::Math::Min(static_cast<int>(color_stop_list.size()), 16);
    gradient_data.num_stops = num_stops;
    for (int i = 0; i < num_stops; i++)
    {
        const auto& stop = color_stop_list[i];
        gradient_data.stops[i].position = stop.position.number;
        Rml::Colourf colorf;
        for (int j = 0; j < 4; j++)
            colorf[j] = static_cast<float>(stop.color[j]) / 255.f;
        gradient_data.stops[i].color = colorf;
    }

    auto ubo = UniformBuffer::Create<GradientData>(gradient_data, m_allocator, m_device, m_descriptor_pool, m_ubo_descriptor_set_layout.get());
    return reinterpret_cast<Rml::CompiledShaderHandle>(ubo.release());
}

void RmlRenderInterface::RenderShader(Rml::CompiledShaderHandle shader_handle, Rml::CompiledGeometryHandle geometry_handle, Rml::Vector2f translation, Rml::TextureHandle texture) {
    auto* geometry = reinterpret_cast<GeometryData*>(geometry_handle);
    const auto* ubo = reinterpret_cast<UniformBuffer*>(shader_handle);
    if (!ubo || !geometry)
        return;

    PushConstants pc{};
    pc.transform = m_projection_matrix * m_transform_matrix;
    pc.translate = translation;
    if (texture)
        pc.texture_id = reinterpret_cast<TextureData*>(texture)->getBindlessIndex();

    const vk::Pipeline pipeline = m_clip_mask_enabled ? m_pipeline_gradient_stencil_clip.get() : m_pipeline_gradient.get();
    m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

    // Bind textures to set 0 and UBO to set 1.
    const std::array sets = { 
        m_textures_descriptor_set.get(),
        ubo->getDescriptorSet()
    };
    m_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipeline_ubo_textures_layout.get(), 0, sets, {});

    m_command_buffer.pushConstants<PushConstants>(m_pipeline_ubo_textures_layout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);

    m_command_buffer.bindVertexBuffers(0, geometry->getBuffer(), { geometry->getVertexOffset() });
    m_command_buffer.bindIndexBuffer(geometry->getBuffer(), geometry->getIndexOffset(), vk::IndexType::eUint32);
    m_command_buffer.drawIndexed(geometry->getNumIndices(), 1, 0, 0, 0);
}

void RmlRenderInterface::ReleaseShader(Rml::CompiledShaderHandle shader_handle)
{
    if (shader_handle)
        delete reinterpret_cast<UniformBuffer*>(shader_handle);
}

#if RMLUI_VULKAN_ENABLE_LAYERS
Rml::LayerHandle RmlRenderInterface::PushLayer() {
    return m_render_layer_manager->PushLayer();
}

void RmlRenderInterface::PopLayer() {
    m_render_layer_manager->PopLayer();
}

void RmlRenderInterface::CompositeLayers(Rml::LayerHandle source_handle, Rml::LayerHandle destination_handle, Rml::BlendMode blend_mode, Rml::Span<const Rml::CompiledFilterHandle> filters) {
    m_render_layer_manager->CompositeLayers(source_handle, destination_handle, blend_mode, filters);
}

Rml::CompiledFilterHandle RmlRenderInterface::CompileFilter(const Rml::String& name, const Rml::Dictionary& parameters) {
    return m_render_layer_manager->CompileFilter(name, parameters);
}

void RmlRenderInterface::ReleaseFilter(Rml::CompiledFilterHandle filter_handle) {
    m_render_layer_manager->ReleaseFilter(filter_handle);
}
#endif
