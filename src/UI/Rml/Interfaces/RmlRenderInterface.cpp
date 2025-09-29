#include "RmlRenderInterface.h"
#include "Log.h"
#include "RmlUi/Core/Core.h"
#include "RmlUi/Core/DecorationTypes.h"
#include "RmlUi/Core/Dictionary.h"
#include "RmlUi/Core/FileInterface.h"
#include "RmlUi/Core/Log.h"
#include "RmlUi/Core/Math.h"
#include "TextureData.h"
#include "UniformBuffer.h"
#include <array>
#include <cmath>
#include "GeometryData.h"

RmlRenderInterface::RmlRenderInterface(
    const vk::Device device, const vk::Queue graphics_queue, const VmaAllocator allocator,
    const vk::CommandPool command_pool, const vk::DescriptorPool descriptor_pool,
    const vk::Format colorFormat, const vk::Format depthFormat)
    : m_device(device), m_queue(graphics_queue), m_allocator(allocator),
      m_command_pool(command_pool), m_descriptor_pool(descriptor_pool),
    m_geometry_pool(m_allocator,  16 * 1024 * 1024, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer)
{
    CreateDescriptors();
    CreatePipelines(colorFormat, depthFormat);

    m_correction_matrix = Rml::Matrix4f::Identity();
    //  flip the Y-axis for Vulkan NDC
    m_correction_matrix[1][1] = -1.0f;
}

void RmlRenderInterface::CreateDescriptors()
{
    //sampler
    const vk::SamplerCreateInfo sampler_ci({}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat);
    m_linear_sampler = m_device.createSamplerUnique(sampler_ci);
    //bindless textures
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

#if RMLUI_VULKAN_ENABLE_SHADERS
    //uniform buffer
    vk::DescriptorSetLayoutBinding ubo_binding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment);
    const vk::DescriptorSetLayoutCreateInfo ubo_layout_ci({}, ubo_binding);
    m_ubo_descriptor_set_layout = m_device.createDescriptorSetLayoutUnique(ubo_layout_ci);
#endif
}


void RmlRenderInterface::CreatePipelines(vk::Format colorFormat, vk::Format depthFormat)
{
    vk::PushConstantRange push_constant_range(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstants));

    m_pipeline_textures_layout = m_device.createPipelineLayoutUnique({{}, m_textures_descriptor_set_layout.get(), push_constant_range});

#if RMLUI_VULKAN_ENABLE_SHADERS
    std::array ubo_textures_layout_sets = {
        m_ubo_descriptor_set_layout.get(),
        m_textures_descriptor_set_layout.get()
    };
    m_pipeline_ubo_textures_layout = m_device.createPipelineLayoutUnique({{}, ubo_textures_layout_sets, push_constant_range});
#endif

    vk::UniqueShaderModule vert_shader = m_device.createShaderModuleUnique({{}, sizeof(shader_vert), reinterpret_cast<const uint32_t*>(shader_vert)});
    vk::UniqueShaderModule frag_shader = m_device.createShaderModuleUnique({{}, sizeof(shader_frag), reinterpret_cast<const uint32_t*>(shader_frag)});
#if RMLUI_VULKAN_ENABLE_SHADERS
    vk::UniqueShaderModule frag_shader_gradient = m_device.createShaderModuleUnique({{}, sizeof(shader_frag_gradient), reinterpret_cast<const uint32_t*>(shader_frag_gradient)});
#endif

    vk::PipelineShaderStageCreateInfo vert_stage_ci({}, vk::ShaderStageFlagBits::eVertex, vert_shader.get(), "main");
    vk::PipelineShaderStageCreateInfo frag_stage_main_ci({}, vk::ShaderStageFlagBits::eFragment, frag_shader.get(), "main");
#if RMLUI_VULKAN_ENABLE_SHADERS
    vk::PipelineShaderStageCreateInfo frag_stage_gradient_ci({}, vk::ShaderStageFlagBits::eFragment, frag_shader_gradient.get(), "main");
#endif

    vk::VertexInputBindingDescription binding_desc(0, sizeof(Rml::Vertex), vk::VertexInputRate::eVertex);
    std::array attribute_descs = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Rml::Vertex, position)),
        vk::VertexInputAttributeDescription(1, 0, vk::Format::eR8G8B8A8Unorm, offsetof(Rml::Vertex, colour)),
        vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Rml::Vertex, tex_coord))
    };
    vk::PipelineVertexInputStateCreateInfo vertex_input_ci({}, binding_desc, attribute_descs);
    vk::PipelineInputAssemblyStateCreateInfo input_assembly_ci({}, vk::PrimitiveTopology::eTriangleList);
    vk::PipelineViewportStateCreateInfo viewport_state_ci({}, 1, nullptr, 1, nullptr);
    vk::PipelineRasterizationStateCreateInfo rasterization_ci({}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise);
    vk::PipelineMultisampleStateCreateInfo multisample_ci({}, vk::SampleCountFlagBits::e1);
#if RMLUI_VULKAN_ENABLE_CLIPPING
    std::array dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor, vk::DynamicState::eStencilReference};
#else
    std::array dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
#endif
    vk::PipelineDynamicStateCreateInfo dynamic_state_ci({}, dynamic_states);
    vk::PipelineRenderingCreateInfo rendering_ci({}, colorFormat, depthFormat, depthFormat);

    vk::PipelineColorBlendAttachmentState color_blend_attachment(true, vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd,vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    vk::PipelineColorBlendStateCreateInfo color_blend_ci({}, false, vk::LogicOp::eCopy, color_blend_attachment);
    vk::PipelineColorBlendAttachmentState no_color_write_attachment;
    no_color_write_attachment.colorWriteMask = {};
    vk::PipelineColorBlendStateCreateInfo no_color_blend_ci({}, false, vk::LogicOp::eCopy, no_color_write_attachment);

    vk::PipelineDepthStencilStateCreateInfo ds_no_stencil({}, false, false);

#if RMLUI_VULKAN_ENABLE_CLIPPING
    vk::PipelineDepthStencilStateCreateInfo ds_stencil_gen{};
    ds_stencil_gen.stencilTestEnable = VK_TRUE;
    ds_stencil_gen.front = vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eReplace, vk::StencilOp::eKeep, vk::CompareOp::eAlways, 0xFF, 0xFF, 1);
    ds_stencil_gen.back = ds_stencil_gen.front;

    vk::PipelineDepthStencilStateCreateInfo ds_stencil_clip{};
    ds_stencil_clip.stencilTestEnable = VK_TRUE;
    ds_stencil_clip.front = vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eEqual, 0xFF, 0xFF, 1);
    ds_stencil_clip.back = ds_stencil_clip.front;

    vk::PipelineDepthStencilStateCreateInfo ds_stencil_incr{};
    ds_stencil_incr.stencilTestEnable = VK_TRUE;
    ds_stencil_incr.front = vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eIncrementAndClamp, vk::StencilOp::eKeep, vk::CompareOp::eEqual, 0xFF, 0xFF, 0);
    ds_stencil_incr.back = ds_stencil_incr.front;

    vk::PipelineDepthStencilStateCreateInfo ds_stencil_zero{};
    ds_stencil_zero.stencilTestEnable = VK_TRUE;
    ds_stencil_zero.front = vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eZero, vk::StencilOp::eKeep, vk::CompareOp::eAlways, 0xFF, 0xFF, 0);
    ds_stencil_zero.back = ds_stencil_zero.front;
#endif

    vk::GraphicsPipelineCreateInfo pipeline_ci;
    pipeline_ci.pNext = &rendering_ci;
    pipeline_ci.pVertexInputState = &vertex_input_ci;
    pipeline_ci.pInputAssemblyState = &input_assembly_ci;
    pipeline_ci.pViewportState = &viewport_state_ci;
    pipeline_ci.pRasterizationState = &rasterization_ci;
    pipeline_ci.pMultisampleState = &multisample_ci;
    pipeline_ci.pDynamicState = &dynamic_state_ci;
    pipeline_ci.stageCount = 2;

    const std::array main_stages = {vert_stage_ci, frag_stage_main_ci};
#if RMLUI_VULKAN_ENABLE_SHADERS
    const std::array gradient_stages = {vert_stage_ci, frag_stage_gradient_ci};
#endif

    std::vector<vk::GraphicsPipelineCreateInfo> pipeline_create_infos;
    std::vector<vk::UniquePipeline*> pipeline_targets;

    pipeline_ci.layout = m_pipeline_textures_layout.get();
    pipeline_ci.pStages = main_stages.data();
    pipeline_ci.pColorBlendState = &color_blend_ci;
    pipeline_ci.pDepthStencilState = &ds_no_stencil;
    pipeline_targets.push_back(&m_pipeline_main);
    pipeline_create_infos.push_back(pipeline_ci);

#if RMLUI_VULKAN_ENABLE_CLIPPING
    pipeline_ci.pColorBlendState = &no_color_blend_ci;
    pipeline_ci.pDepthStencilState = &ds_stencil_gen;
    pipeline_targets.push_back(&m_pipeline_stencil_gen);
    pipeline_create_infos.push_back(pipeline_ci);

    pipeline_ci.pDepthStencilState = &ds_stencil_incr;
    pipeline_targets.push_back(&m_pipeline_stencil_incr);
    pipeline_create_infos.push_back(pipeline_ci);

    pipeline_ci.pDepthStencilState = &ds_stencil_zero;
    pipeline_targets.push_back(&m_pipeline_stencil_zero);
    pipeline_create_infos.push_back(pipeline_ci);

    pipeline_ci.pColorBlendState = &color_blend_ci;
    pipeline_ci.pDepthStencilState = &ds_stencil_clip;
    pipeline_targets.push_back(&m_pipeline_stencil_clip);
    pipeline_create_infos.push_back(pipeline_ci);
#endif

#if RMLUI_VULKAN_ENABLE_SHADERS
    pipeline_ci.layout = m_pipeline_ubo_textures_layout.get();
    pipeline_ci.pStages = gradient_stages.data();
    pipeline_ci.pColorBlendState = &color_blend_ci;
    pipeline_ci.pDepthStencilState = &ds_no_stencil;
    pipeline_targets.push_back(&m_pipeline_gradient);
    pipeline_create_infos.push_back(pipeline_ci);

#if RMLUI_VULKAN_ENABLE_CLIPPING
    pipeline_ci.pDepthStencilState = &ds_stencil_clip;
    pipeline_targets.push_back(&m_pipeline_gradient_stencil_clip);
    pipeline_create_infos.push_back(pipeline_ci);
#endif

#endif

    auto result = m_device.createGraphicsPipelinesUnique({}, pipeline_create_infos);

    if (result.result != vk::Result::eSuccess)
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create UI pipelines: %s", vk::to_string(result.result).c_str());
    else
        for (size_t i = 0; i < result.value.size(); ++i)
            *pipeline_targets[i] = std::move(result.value[i]);
}


RmlRenderInterface::~RmlRenderInterface()
{
    if (m_device)
    {
        try
        {
            m_device.waitIdle();

            while (!m_destruction_queue.empty()) {
                auto& [fence, resources] = m_destruction_queue.front();

                if (m_device.getFenceStatus(fence) == vk::Result::eSuccess) {
                    for (auto* res : resources) {
                        // Only recycle bindless indices if this is a TextureData
                        if (auto* tex = dynamic_cast<TextureData*>(res))
                            if (tex->getBindlessIndex() != static_cast<uint32_t>(-1))
                                m_free_texture_indices.push_back(tex->getBindlessIndex());

                        delete res; // safe to delete now
                    }
                    m_destruction_queue.pop_front();
                } else
                    break; // stop if oldest fence hasn't signaled yet
            }

            m_registered_textures.clear();
        }
        catch (const vk::SystemError& err)
        {
            Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan device waitIdle failed in destructor: %s", err.what());
        }
    }
}

Rml::CompiledGeometryHandle RmlRenderInterface::CompileGeometry(
    const Rml::Span<const Rml::Vertex> vertices,
    const Rml::Span<const int> indices)
{
    // Now just create a new GeometryData object that uses the pool.
    auto* geom = new GeometryData(m_geometry_pool, vertices, indices);
    return reinterpret_cast<Rml::CompiledGeometryHandle>(geom);
}
    

void RmlRenderInterface::ReleaseGeometry(const Rml::CompiledGeometryHandle handle)
{
    if (!handle || !m_current_frame_fence) return;

    auto* geom = reinterpret_cast<GeometryData*>(handle);
    m_destruction_queue.back().second.push_back(geom);
}

Rml::TextureHandle RmlRenderInterface::GenerateTexture(const Rml::Span<const Rml::byte> source_data, const Rml::Vector2i source_dimensions)
{
    if (m_free_texture_indices.empty())
    {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create texture: no free bindless indices available.");
        return {};
    }

    auto [texture, staging_Buffer] = TextureData::Create(m_allocator, m_device, m_command_buffer, source_data, source_dimensions, m_linear_sampler.get());

    m_destruction_queue.back().second.push_back(staging_Buffer);

    // Set bindless Index
    const uint32_t id = m_free_texture_indices.back();
    m_free_texture_indices.pop_back();
    texture->setBindlessIndex(id);

    vk::DescriptorImageInfo image_info(texture->getSampler(), texture->getImageView(), vk::ImageLayout::eShaderReadOnlyOptimal);
    const vk::WriteDescriptorSet write(m_textures_descriptor_set.get(), 0, id, vk::DescriptorType::eCombinedImageSampler, image_info);
    m_device.updateDescriptorSets(write, {});

    // Return the raw pointer handle to RmlUi.
    return reinterpret_cast<Rml::TextureHandle>(texture);
}

void RmlRenderInterface::ReleaseTexture(const Rml::TextureHandle handle)
{
    if (handle && m_current_frame_fence)
    {
        auto* texture = reinterpret_cast<TextureData*>(handle);
        m_destruction_queue.back().second.push_back(texture);
    }
}

void RmlRenderInterface::registerVulkanTexture(const Rml::String& name, const vk::ImageView image_view, const Rml::Vector2i dimensions)
{
    if (m_registered_textures.contains(name))
    {
        Rml::Log::Message(Rml::Log::LT_WARNING, "A Vulkan texture with the name '%s' is already registered. Ignoring request.", name.c_str());
        return;
    }

    if (m_free_texture_indices.empty())
    {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to register Vulkan texture '%s': no free bindless indices available.", name.c_str());
        return;
    }

    auto texture_ptr = new TextureData(m_allocator, m_device, image_view);

    const uint32_t id = m_free_texture_indices.back();
    m_free_texture_indices.pop_back();
    texture_ptr->setBindlessIndex(id);

    vk::DescriptorImageInfo image_info(texture_ptr->getSampler(), texture_ptr->getImageView(), vk::ImageLayout::eShaderReadOnlyOptimal);
    const vk::WriteDescriptorSet write(m_textures_descriptor_set.get(), 0, id, vk::DescriptorType::eCombinedImageSampler, image_info);
    m_device.updateDescriptorSets(write, {});

    Rml::TextureHandle handle = reinterpret_cast<Rml::TextureHandle>(texture_ptr);
    m_registered_textures[name] = {handle, dimensions};
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
            // Correct for TGA's vertical flip if necessary
            const int write_idx = ((!(header.imageDescriptor & 32) ? (header.height - 1 - y) : y) * header.width + x) * 4;
            // Assign straight RGBA values first
            image_dest[write_idx] = image_src[read_idx + 2];
            image_dest[write_idx + 1] = image_src[read_idx + 1];
            image_dest[write_idx + 2] = image_src[read_idx];
            if (color_mode == 4) // premultiply color by alpha
            {
                const Rml::byte alpha = image_src[read_idx + 3];
                image_dest[write_idx] = static_cast<Rml::byte>((image_dest[write_idx] * alpha) / 255);
                image_dest[write_idx + 1] = static_cast<Rml::byte>((image_dest[write_idx + 1] * alpha) / 255);
                image_dest[write_idx + 2] = static_cast<Rml::byte>((image_dest[write_idx + 2] * alpha) / 255);
                image_dest[write_idx + 3] = alpha;
            }
            else
                image_dest[write_idx + 3] = 255;
        }
    }

    return GenerateTexture({image_dest.get(), image_size}, texture_dimensions);
}

void RmlRenderInterface::SetTransform(const Rml::Matrix4f* transform)
{
    m_transform_enabled = (transform != nullptr);
    m_transform_matrix = transform ? *transform : Rml::Matrix4f::Identity();
}

void RmlRenderInterface::beginFrame(const vk::CommandBuffer command_buffer, const vk::Image target_image, const vk::ImageView target_image_view, const vk::ImageView depthImageView, vk::Extent2D target_extent, vk::Fence in_flight_fence)
{
    m_command_buffer = command_buffer;
    m_target_extent = target_extent;
    m_target_image_view = target_image_view;
    m_target_image = target_image;

    // Process destruction queue BEFORE setting up new frame
    while (!m_destruction_queue.empty()) {
        auto& [fence, resources] = m_destruction_queue.front();

        if (m_device.getFenceStatus(fence) == vk::Result::eSuccess) {
            for (auto* res : resources) {
                if (res->type == GpuResource::GpuResourceType::Texture)
                {
                    auto* tex = static_cast<TextureData*>(res);
                    if (tex->getBindlessIndex() != static_cast<uint32_t>(-1))
                        m_free_texture_indices.push_back(tex->getBindlessIndex());
                }
                delete res;
            }
            m_destruction_queue.pop_front();
        } else break;
    }

    // Now set up new frame
    m_current_frame_fence = in_flight_fence;
    m_destruction_queue.emplace_back(m_current_frame_fence, std::vector<GpuResource*>{});
    
    // Update projection matrix
    const float w = static_cast<float>(target_extent.width);
    const float h = static_cast<float>(target_extent.height);
    m_projection_matrix = m_correction_matrix * Rml::Matrix4f::ProjectOrtho(0.0f, w, h, 0.0f, -10000.0f, 10000.0f);

    vk::RenderingAttachmentInfo color_attachment_info(
        target_image_view, vk::ImageLayout::eColorAttachmentOptimal,
        {}, {}, {},
        vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
        vk::ClearColorValue(std::array{0.0f, 0.0f, 0.0f, 0.0f})
    );
    const vk::RenderingAttachmentInfo depth_stencil_attachment_info(
        depthImageView, vk::ImageLayout::eDepthStencilAttachmentOptimal,
        {}, {}, {},
        vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
        vk::ClearDepthStencilValue(1.0f, 0)
    );
    const vk::RenderingInfo rendering_info({}, {{0, 0}, target_extent}, 1, 0, color_attachment_info, &depth_stencil_attachment_info, &depth_stencil_attachment_info);

    command_buffer.beginRendering(rendering_info);

    const vk::Viewport viewport(0.0f, 0.0f, w, h, 0.0f, 1.0f);
    command_buffer.setViewport(0, viewport);
    const vk::Rect2D scissor({0, 0}, target_extent);
    command_buffer.setScissor(0, scissor);
    m_current_scissor = scissor;

    // Reset other per-frame states
#if RMLUI_VULKAN_ENABLE_CLIPPING
    m_clip_mask_enabled = false;
#endif
    m_transform_enabled = false;
    m_transform_matrix = Rml::Matrix4f::Identity();
}

void RmlRenderInterface::endFrame()
{
    m_command_buffer.endRendering();
}

void RmlRenderInterface::renderCall(const GeometryData* geometry)
{
    const vk::Buffer vertexBuffer = m_geometry_pool.getBuffer();
    const vk::DeviceSize vertexOffset = geometry->getVertexOffset();
    m_command_buffer.bindVertexBuffers(0, 1, &vertexBuffer, &vertexOffset);
    const vk::DeviceSize indexOffset = geometry->getIndexOffset();
    m_command_buffer.bindIndexBuffer(vertexBuffer, indexOffset, vk::IndexType::eUint32);
    m_command_buffer.drawIndexed(geometry->getNumIndices(), 1, 0, 0, 0);
}

void RmlRenderInterface::RenderGeometry(Rml::CompiledGeometryHandle geometry_handle, Rml::Vector2f translation, Rml::TextureHandle texture)
{
    auto* geometry = reinterpret_cast<GeometryData*>(geometry_handle);
    if (!geometry) return;

    PushConstants pc{};
    pc.transform = m_projection_matrix * m_transform_matrix;
    pc.translate = translation;
    if (texture)
        pc.texture_id = reinterpret_cast<TextureData*>(texture)->getBindlessIndex();

#if RMLUI_VULKAN_ENABLE_CLIPPING
    const vk::Pipeline pipeline = m_clip_mask_enabled ? m_pipeline_stencil_clip.get() : m_pipeline_main.get();

#else
    const vk::Pipeline pipeline = m_pipeline_main.get();
#endif
    m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

#if RMLUI_VULKAN_ENABLE_CLIPPING
    if (m_clip_mask_enabled)
        m_command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, m_stencil_level);
#endif
    
    m_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipeline_textures_layout.get(), 0, m_textures_descriptor_set.get(), {});
    m_command_buffer.pushConstants<PushConstants>(m_pipeline_textures_layout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
    renderCall(geometry);
}

void RmlRenderInterface::SetScissorRegion(const Rml::Rectanglei region)
{
    if (!m_scissor_enabled) return;

#if RMLUI_VULKAN_ENABLE_CLIPPING
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
#endif
    {
#if RMLUI_VULKAN_ENABLE_CLIPPING
        EnableClipMask(false);
#endif
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
#if RMLUI_VULKAN_ENABLE_CLIPPING
        EnableClipMask(false);
#endif
        m_command_buffer.setScissor(0, m_current_scissor);
    }
}

#if RMLUI_VULKAN_ENABLE_CLIPPING

void RmlRenderInterface::EnableClipMask(const bool enable)
{
    m_clip_mask_enabled = enable;
    if (!enable)
        m_stencil_level = 0;
}

void RmlRenderInterface::RenderToClipMask(Rml::ClipMaskOperation operation, const Rml::CompiledGeometryHandle handle, const Rml::Vector2f translation)
{
    const auto* geometry = reinterpret_cast<GeometryData*>(handle);
    if (!geometry) return;

    PushConstants constants{};
    constants.transform = m_projection_matrix * m_transform_matrix;
    constants.translate = translation;
    m_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipeline_textures_layout.get(), 0, m_textures_descriptor_set.get(), {});

    switch (operation)
    {
    case Rml::ClipMaskOperation::Set:
        {
            m_stencil_level = 1;
            m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline_stencil_gen.get());
            m_command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, m_stencil_level);
            m_command_buffer.pushConstants<PushConstants>(m_pipeline_textures_layout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, constants);
            // Draw the mask geometry
            renderCall(geometry);
            break;
        }

    case Rml::ClipMaskOperation::SetInverse:
        {
            m_stencil_level = 1;
            // 1. Fill the entire stencil buffer with '1'
            m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline_stencil_gen.get());
            m_command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, m_stencil_level);

            const auto* quad_geo = reinterpret_cast<GeometryData*>(m_fullscreen_quad);
            const PushConstants quad_pc = {Rml::Matrix4f::Identity(), {0, 0}, -1};
            // Push quad constants
            m_command_buffer.pushConstants<PushConstants>(m_pipeline_textures_layout.get(),vk::ShaderStageFlagBits::eVertex,0,quad_pc);
            // Render quad geometry
            renderCall(quad_geo);
            // 2. Punch a hole (write '0') where the user geometry is
            m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline_stencil_zero.get());
            m_command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, 0);
            // Push user geometry constants
            m_command_buffer.pushConstants<PushConstants>(m_pipeline_textures_layout.get(),vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,0,constants);
            // Render user geometry
            renderCall(geometry);
            break;
        }

    case Rml::ClipMaskOperation::Intersect:
        {
            m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline_stencil_incr.get());
            m_command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, m_stencil_level);
            m_stencil_level++;
            // Push constants
            m_command_buffer.pushConstants<PushConstants>(m_pipeline_textures_layout.get(),vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,0,constants);
            // Render user geometry
            renderCall(geometry);
            break;
        }
    }
}
#endif

#if RMLUI_VULKAN_ENABLE_SHADERS
Rml::CompiledShaderHandle RmlRenderInterface::CompileShader(const Rml::String& name, const Rml::Dictionary& parameters)
{
    GradientData gradient_data{};

    if (name == "linear-gradient")
    {
        gradient_data.gradient_function = GRADIENT_LINEAR;
        gradient_data.repeating = Rml::Get(parameters, "repeating", false);
        gradient_data.p = Rml::Get(parameters, "p0", Rml::Vector2f(0.f));
        Rml::Vector2f p1 = Rml::Get(parameters, "p1", Rml::Vector2f(1.f, 0.f));
        gradient_data.v = p1 - gradient_data.p;
        if (gradient_data.v.SquaredMagnitude() < 1e-6f)
            gradient_data.v = {1.f, 0.f};
    }
    else if (name == "radial-gradient")
    {
        gradient_data.gradient_function = GRADIENT_RADIAL;
        gradient_data.repeating = Rml::Get(parameters, "repeating", false);
        gradient_data.p = Rml::Get(parameters, "center", Rml::Vector2f(0.5f));
        Rml::Vector2f radius_uv = Rml::Get(parameters, "radius", Rml::Vector2f(0.5f));
        radius_uv.x = Rml::Math::Max(radius_uv.x, 1e-6f);
        radius_uv.y = Rml::Math::Max(radius_uv.y, 1e-6f);
        gradient_data.v = Rml::Vector2f(1.0f) / radius_uv;
    }
    else if (name == "conic-gradient")
    {
        gradient_data.gradient_function = GRADIENT_CONIC;
        gradient_data.repeating = Rml::Get(parameters, "repeating", false);
        // Expects 'center' directly in UV space [0,1]. Default is the center of the element.
        gradient_data.p = Rml::Get(parameters, "center", Rml::Vector2f(0.5f));
        // Expects 'angle' in radians
        const float angle_rad = Rml::Get(parameters, "angle", 0.0f);
        gradient_data.v = {Rml::Math::Cos(angle_rad), Rml::Math::Sin(angle_rad)};
    }
    else
    {
        Rml::Log::Message(Rml::Log::LT_WARNING, "Unsupported shader type '%s'.", name.c_str());
        return {};
    }

    const auto it = parameters.find("color_stop_list");
    if (it == parameters.end())
    {
        Rml::Log::Message(Rml::Log::LT_ERROR, "No 'color_stop_list' found for gradient shader '%s'.", name.c_str());
        return {};
    }
    const auto& color_stop_list = it->second.GetReference<Rml::ColorStopList>();
    const int num_stops = Rml::Math::Min(static_cast<int>(color_stop_list.size()), MAX_STOPS);
    gradient_data.num_stops = num_stops;
    for (int i = 0; i < num_stops; i++)
    {
        const auto& stop = color_stop_list[i];
        gradient_data.stops[i].position = stop.position.number;
        gradient_data.stops[i].color = {
            static_cast<float>(stop.color.red) / 255.f,
            static_cast<float>(stop.color.green) / 255.f,
            static_cast<float>(stop.color.blue) / 255.f,
            static_cast<float>(stop.color.alpha) / 255.f
        };
    }

    // Create the UBO for the shader
    return reinterpret_cast<Rml::CompiledShaderHandle>(UniformBuffer::Create<GradientData>(gradient_data, m_allocator, m_device, m_descriptor_pool, m_ubo_descriptor_set_layout.get()));
}

void RmlRenderInterface::RenderShader(Rml::CompiledShaderHandle shader_handle, Rml::CompiledGeometryHandle geometry_handle, Rml::Vector2f translation, Rml::TextureHandle texture)
{
    const auto* geometry = reinterpret_cast<GeometryData*>(geometry_handle);
    const auto* ubo = reinterpret_cast<UniformBuffer*>(shader_handle);
    if (!ubo || !geometry) return;

    PushConstants pc{};
    pc.transform = m_projection_matrix * m_transform_matrix;
    pc.translate = translation;
    if (texture)
        pc.texture_id = reinterpret_cast<TextureData*>(texture)->getBindlessIndex();

#if RMLUI_VULKAN_ENABLE_CLIPPING
    const vk::Pipeline pipeline = m_clip_mask_enabled ? m_pipeline_gradient_stencil_clip.get() : m_pipeline_gradient.get();
#else
    const vk::Pipeline pipeline = m_pipeline_gradient.get();
#endif
    m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

    const std::array sets = {
        ubo->getDescriptorSet(),
        m_textures_descriptor_set.get()
    };
    m_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipeline_ubo_textures_layout.get(), 0, sets, {});
    m_command_buffer.pushConstants<PushConstants>(m_pipeline_ubo_textures_layout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
    renderCall(geometry);
}

void RmlRenderInterface::ReleaseShader(Rml::CompiledShaderHandle shader_handle)
{
    if (shader_handle) delete reinterpret_cast<UniformBuffer*>(shader_handle);
}
#endif