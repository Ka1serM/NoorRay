#include "RmlRenderInterface.h"
#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/FileInterface.h>
#include <RmlUi/Core/Log.h>
#include <RmlUi/Core/Math.h>
#include <array>
#include "Log.h"
#include "RmlUi/Core/DecorationTypes.h"

RmlRenderInterface::RmlRenderInterface(
    const vk::Device device,
    const vk::Queue graphics_queue,
    const VmaAllocator allocator,
    const vk::CommandPool command_pool,
    const vk::DescriptorPool descriptor_pool,
    const vk::Format colorFormat,
    const vk::Format depthFormat
) : m_device(device), 
    m_graphics_queue(graphics_queue),
    m_allocator(allocator),
    m_utility_command_pool(command_pool),
    m_descriptor_pool(descriptor_pool)
{
   CreateDescriptors();
   CreatePipelines(colorFormat, depthFormat);
}

RmlRenderInterface::~RmlRenderInterface() {
    if (m_device) {
       try {
          m_device.waitIdle();

           // Clear registered textures first, which queues them for destruction
           m_registered_textures.clear();

           for (auto const& [fence, geometries] : m_destruction_map_geometry)
               for (const auto& geometry : geometries)
                   vmaDestroyBuffer(m_allocator, geometry->buffer, geometry->allocation);
           m_destruction_map_geometry.clear();

           for (auto const& [fence, textures] : m_destruction_map_textures)
               for (auto& texture : textures)
                   DestroyTexture(texture.get());
           m_destruction_map_textures.clear();
           
       } catch (const vk::SystemError& err) {
          Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan device waitIdle failed in destructor: %s", err.what());
       }
    }
}

void RmlRenderInterface::beginFrame(const vk::CommandBuffer command_buffer, const vk::ImageView target_image_view, const vk::ImageView depthImageView, vk::Extent2D target_extent, vk::Fence in_flight_fence) {
    m_current_frame_fence = in_flight_fence;
    m_command_buffer = command_buffer;
    
    // Clean up textures from past frames whose fences are now signaled
    if(m_destruction_map_textures.contains(m_current_frame_fence)) {
        for (auto& texture : m_destruction_map_textures.at(m_current_frame_fence))
            DestroyTexture(texture.get());
        m_destruction_map_textures.erase(m_current_frame_fence);
    }
    // Clean up geometry from past frames whose fences are now signaled
    if(m_destruction_map_geometry.contains(m_current_frame_fence)) {
        for (const auto& geometry : m_destruction_map_geometry.at(m_current_frame_fence))
            vmaDestroyBuffer(m_allocator, geometry->buffer, geometry->allocation);
        m_destruction_map_geometry.erase(m_current_frame_fence);
    }
    
    vk::RenderingAttachmentInfo color_attachment_info(
        target_image_view, vk::ImageLayout::eColorAttachmentOptimal, 
        {}, {}, {}, 
        vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, 
        vk::ClearColorValue(std::array{0.0f, 0.0f, 0.0f, 1.0f})
    );
    
    const vk::RenderingAttachmentInfo depth_stencil_attachment_info(
        depthImageView, vk::ImageLayout::eDepthStencilAttachmentOptimal,
        {}, {}, {},
        vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
        vk::ClearDepthStencilValue(1.0f, 0)
    );

    const vk::RenderingInfo rendering_info({}, 
        {{0, 0}, target_extent}, 
        1, 0, 
        color_attachment_info, 
        &depth_stencil_attachment_info,     // pDepthAttachment
        &depth_stencil_attachment_info      // pStencilAttachment
    );

    command_buffer.beginRendering(rendering_info);

    const vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(target_extent.width), static_cast<float>(target_extent.height), 0.0f, 1.0f);
    command_buffer.setViewport(0, viewport);
    
    const vk::Rect2D scissor({0, 0}, target_extent);
    command_buffer.setScissor(0, scissor);
    m_current_scissor = scissor;

    // Update projection matrix
    const float w = static_cast<float>(target_extent.width);
    const float h = static_cast<float>(target_extent.height);
    m_projection_matrix = Rml::Matrix4f::ProjectOrtho(0.0f, w, h, 0.0f, -10000.0f, 10000.0f);
    
    Rml::Matrix4f correction = Rml::Matrix4f::Identity();
    //  flip the Y-axis for Vulkan NDC
    correction[1][1] = -1.0f;
    m_projection_matrix = correction * m_projection_matrix;
    
    m_clip_mask_enabled = false;
    m_transform_enabled = false;
    m_transform_matrix = Rml::Matrix4f::Identity();
}

Rml::CompiledGeometryHandle RmlRenderInterface::CompileGeometry(const Rml::Span<const Rml::Vertex> vertices, const Rml::Span<const int> indices) {
    auto geometry = std::make_unique<GeometryData>();
    geometry->num_indices = static_cast<int>(indices.size());

    const vk::DeviceSize vertices_size = vertices.size() * sizeof(Rml::Vertex);
    const vk::DeviceSize indices_size = indices.size() * sizeof(int);
    const vk::DeviceSize total_size = vertices_size + indices_size;

    geometry->vertex_offset = 0;
    geometry->index_offset = vertices_size;

    const auto buffer_ci = static_cast<VkBufferCreateInfo>(vk::BufferCreateInfo({}, total_size, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer));
    constexpr VmaAllocationCreateInfo alloc_ci = {VMA_ALLOCATION_CREATE_MAPPED_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU};
    
    VmaAllocationInfo alloc_info = {};
    VkBuffer raw_buffer;
    vmaCreateBuffer(m_allocator, &buffer_ci, &alloc_ci, &raw_buffer, &geometry->allocation, &alloc_info);
    geometry->buffer = raw_buffer;

    memcpy(alloc_info.pMappedData, vertices.data(), vertices_size);
    memcpy(static_cast<std::byte*>(alloc_info.pMappedData) + vertices_size, indices.data(), indices_size);

    return reinterpret_cast<Rml::CompiledGeometryHandle>(geometry.release());
}

void RmlRenderInterface::RenderGeometry(Rml::CompiledGeometryHandle geometry_handle, Rml::Vector2f translation, Rml::TextureHandle texture)
{
    auto* geometry = reinterpret_cast<GeometryData*>(geometry_handle);

    PushConstants pc{};
    pc.transform = m_projection_matrix * m_transform_matrix;
    pc.translate = translation;
    pc.texture_id = texture ? reinterpret_cast<TextureData*>(texture)->bindless_index : -1;

    // Bind the regular pipeline
    const vk::Pipeline pipeline = m_clip_mask_enabled ? m_pipeline_stencil_clip.get() : m_pipeline_main.get();
    m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

    m_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipeline_layout.get(), 0, m_bindless_descriptor_set,{});

    m_command_buffer.pushConstants<PushConstants>(m_pipeline_layout.get(),vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);

    m_command_buffer.bindVertexBuffers(0, geometry->buffer, { geometry->vertex_offset });
    m_command_buffer.bindIndexBuffer(geometry->buffer, geometry->index_offset, vk::IndexType::eUint32);
    m_command_buffer.drawIndexed(geometry->num_indices, 1, 0, 0, 0);
}

void RmlRenderInterface::ReleaseGeometry(const Rml::CompiledGeometryHandle geometry_handle) {
    if (geometry_handle && m_current_frame_fence)
        m_destruction_map_geometry[m_current_frame_fence].emplace_back(reinterpret_cast<GeometryData*>(geometry_handle));
}

void RmlRenderInterface::registerVulkanTexture(const Rml::String& name, const vk::ImageView image_view, const Rml::Vector2i dimensions) {
    if (m_registered_textures.contains(name)) {
        Rml::Log::Message(Rml::Log::LT_WARNING, "A Vulkan texture with the name '%s' is already registered. Ignoring request.", name.c_str());
        return;
    }

    if (const Rml::TextureHandle handle = CreateTextureHandleForView(image_view)) {
        m_registered_textures[name] = { handle, dimensions };
    } else {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create texture handle for Vulkan texture '%s'.", name.c_str());
    }
}

void RmlRenderInterface::unregisterVulkanTexture(const Rml::String& name) {
    const auto it = m_registered_textures.find(name);
    if (it != m_registered_textures.end()) {
        ReleaseTexture(it->second.first);
        m_registered_textures.erase(it);
    }
}

Rml::TextureHandle RmlRenderInterface::CreateTextureHandleForView(const vk::ImageView image_view) {
    auto texture = std::make_unique<TextureData>();
    texture->image_view_raw = image_view; // Store the non-owned handle

    vk::SamplerCreateInfo sampler_info({}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge);
    // Registered textures get a unique sampler. We own this sampler.
    texture->sampler = m_device.createSampler(sampler_info);

    if (m_free_texture_indices.empty()) {
        Rml::Log::Message(Rml::Log::LT_ERROR, "No free bindless indices for Vulkan texture!");
        m_device.destroySampler(texture->sampler); // Clean up the sampler we just made
        return {};
    }

    const uint32_t id = m_free_texture_indices.back();
    m_free_texture_indices.pop_back();
    texture->bindless_index = id;

    vk::DescriptorImageInfo image_info(texture->sampler, texture->getImageView(), vk::ImageLayout::eShaderReadOnlyOptimal);
    const vk::WriteDescriptorSet write(m_bindless_descriptor_set, 0, texture->bindless_index, vk::DescriptorType::eCombinedImageSampler, image_info);
    m_device.updateDescriptorSets(write, {});

    return reinterpret_cast<Rml::TextureHandle>(texture.release());
}


Rml::TextureHandle RmlRenderInterface::LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source) {
    LOG_INFO("Loading Texture" + source);
    
    if (source.rfind("vulkan://", 0) == 0) {
        const Rml::String texture_name = source.substr(9);
        const auto it = m_registered_textures.find(texture_name);
        
        if (it != m_registered_textures.end()) {
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

    if (buffer_size <= sizeof(TGAHeader)) {
        file_interface->Close(file_handle);
        return {};
    }

    const auto buffer = std::unique_ptr<Rml::byte[]>(new Rml::byte[buffer_size]);
    file_interface->Read(buffer.get(), buffer_size, file_handle);
    file_interface->Close(file_handle);

    TGAHeader header{};
    memcpy(&header, buffer.get(), sizeof(TGAHeader));

    if (header.dataType != 2 || (header.bitsPerPixel != 24 && header.bitsPerPixel != 32)) {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Unsupported TGA: %s. Only 24/32bit uncompressed supported.", source.c_str());
        return {};
    }

    texture_dimensions = {header.width, header.height};
    const int color_mode = header.bitsPerPixel / 8;
    const size_t image_size = static_cast<size_t>(header.width) * header.height * 4;
    auto image_dest = std::unique_ptr<Rml::byte[]>(new Rml::byte[image_size]);
    const Rml::byte* image_src = buffer.get() + sizeof(TGAHeader);

    for (int y = 0; y < header.height; ++y) {
        for (int x = 0; x < header.width; ++x) {
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
                image_dest[write_idx]     = Rml::byte( (image_dest[write_idx]     * alpha) / 255 );
                image_dest[write_idx + 1] = Rml::byte( (image_dest[write_idx + 1] * alpha) / 255 );
                image_dest[write_idx + 2] = Rml::byte( (image_dest[write_idx + 2] * alpha) / 255 );
                image_dest[write_idx + 3] = alpha;
            }
            else
                image_dest[write_idx + 3] = 255;
        }
    }

    return GenerateTexture({image_dest.get(), image_size}, texture_dimensions);
}

Rml::TextureHandle RmlRenderInterface::GenerateTexture(const Rml::Span<const Rml::byte> source_data, const Rml::Vector2i source_dimensions) {
    auto texture = std::make_unique<TextureData>();
    texture->sampler = m_linear_sampler.get(); // Generated textures use the shared sampler

    if (m_free_texture_indices.empty()) {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create texture: no free bindless indices available.");
        return {};
    }
    
    const uint32_t id = m_free_texture_indices.back();
    m_free_texture_indices.pop_back();
    texture->bindless_index = id;
    
    vk::DeviceSize image_size = source_data.size();
    vk::Extent3D extent(source_dimensions.x, source_dimensions.y, 1);
    
    auto staging_buffer_ci = static_cast<VkBufferCreateInfo>(vk::BufferCreateInfo({}, image_size, vk::BufferUsageFlagBits::eTransferSrc));
    VmaAllocationCreateInfo staging_alloc_ci = {VMA_ALLOCATION_CREATE_MAPPED_BIT, VMA_MEMORY_USAGE_CPU_ONLY};
    VkBuffer staging_buffer;
    VmaAllocation staging_allocation;
    VmaAllocationInfo staging_alloc_info;
    vmaCreateBuffer(m_allocator, &staging_buffer_ci, &staging_alloc_ci, &staging_buffer, &staging_allocation, &staging_alloc_info);
    memcpy(staging_alloc_info.pMappedData, source_data.data(), image_size);

    auto image_ci = static_cast<VkImageCreateInfo>(vk::ImageCreateInfo({}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Unorm, extent, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled));
    VmaAllocationCreateInfo image_alloc_ci = {0, VMA_MEMORY_USAGE_GPU_ONLY};
    VkImage raw_image;
    vmaCreateImage(m_allocator, &image_ci, &image_alloc_ci, &raw_image, &texture->allocation, nullptr);
    texture->image = raw_image; // We own this image

    vk::CommandBufferAllocateInfo alloc_info(m_utility_command_pool, vk::CommandBufferLevel::ePrimary, 1);
    vk::UniqueCommandBuffer cmd = std::move(m_device.allocateCommandBuffersUnique(alloc_info)[0]);
    cmd->begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    vk::ImageMemoryBarrier to_transfer({}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, texture->image, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    cmd->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, nullptr, nullptr, to_transfer);

    vk::BufferImageCopy copy_region(0, 0, 0, { vk::ImageAspectFlagBits::eColor, 0, 0, 1 }, {}, extent);
    cmd->copyBufferToImage(staging_buffer, texture->image, vk::ImageLayout::eTransferDstOptimal, copy_region);

    vk::ImageMemoryBarrier to_shader_read(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, texture->image, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    cmd->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, to_shader_read);

    cmd->end();

    m_graphics_queue.submit(vk::SubmitInfo({}, {}, *cmd), {});
    m_graphics_queue.waitIdle();    

    vmaDestroyBuffer(m_allocator, staging_buffer, staging_allocation);
    
    vk::ImageViewCreateInfo view_ci({}, texture->image, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Unorm, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    texture->owned_image_view = m_device.createImageViewUnique(view_ci); // We own this view
    
    vk::DescriptorImageInfo image_info(texture->sampler, texture->getImageView(), vk::ImageLayout::eShaderReadOnlyOptimal);
    vk::WriteDescriptorSet write(m_bindless_descriptor_set, 0, texture->bindless_index, vk::DescriptorType::eCombinedImageSampler, image_info);
    m_device.updateDescriptorSets(write, {});

    return reinterpret_cast<Rml::TextureHandle>(texture.release());
}

void RmlRenderInterface::ReleaseTexture(const Rml::TextureHandle texture_handle) {
    if (texture_handle && m_current_frame_fence)
        m_destruction_map_textures[m_current_frame_fence].emplace_back(reinterpret_cast<TextureData*>(texture_handle));
}

void RmlRenderInterface::SetScissorRegion(const Rml::Rectanglei region) {
    if (!m_scissor_enabled) return;
    
    if (m_transform_enabled) {
        // A transformed scissor region is a prime use case for the clip mask!
        EnableClipMask(true);

        // Create geometry for the scissor rectangle
        Rml::Vertex vertices[4];
        vertices[0].position = {static_cast<float>(region.Left()), static_cast<float>(region.Top())};
        vertices[1].position = {static_cast<float>(region.Right()), static_cast<float>(region.Top())};
        vertices[2].position = {static_cast<float>(region.Right()), static_cast<float>(region.Bottom())};
        vertices[3].position = {static_cast<float>(region.Left()), static_cast<float>(region.Bottom())};
        int indices[] = {0, 1, 2, 0, 2, 3};
        
        // Compile, render to the mask, and release the temporary geometry
        if (const auto handle = CompileGeometry({vertices, 4}, {indices, 6})) {
            RenderToClipMask(Rml::ClipMaskOperation::Set, handle, {});
            ReleaseGeometry(handle);
        }
    } else {
        // For simple, axis-aligned scissors, we don't need the stencil buffer.
        // And we must ensure the clip mask is off.
        EnableClipMask(false);
        
        vk::Rect2D scissor;
        scissor.offset.x = Rml::Math::Max(0, region.Left());
        scissor.offset.y = Rml::Math::Max(0, region.Top());
        scissor.extent.width = region.Width();
        scissor.extent.height = region.Height();
        m_command_buffer.setScissor(0, scissor);
    }
}

// Also update EnableScissorRegion to use the new flag
void RmlRenderInterface::EnableScissorRegion(const bool enable) {
    m_scissor_enabled = enable;
    if (!enable) {
        // If scissoring is disabled entirely, the clip mask should also be disabled.
        EnableClipMask(false);
        m_command_buffer.setScissor(0, m_current_scissor);
    }
}

void RmlRenderInterface::SetTransform(const Rml::Matrix4f* transform) {
    m_transform_enabled = (transform != nullptr);
    m_transform_matrix = transform ? *transform : Rml::Matrix4f::Identity();
}

void RmlRenderInterface::endFrame() const {
    if(m_command_buffer)
        m_command_buffer.endRendering();
}
void RmlRenderInterface::CreateDescriptors() {
    constexpr vk::SamplerCreateInfo sampler_ci({}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat);

    m_linear_sampler = m_device.createSamplerUnique(sampler_ci);
    
    vk::DescriptorSetLayoutBinding binding(0, vk::DescriptorType::eCombinedImageSampler, kMaxBindlessTextures, vk::ShaderStageFlagBits::eFragment);
    vk::DescriptorBindingFlags binding_flags = vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind;
    const vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_ci(binding_flags);
    vk::DescriptorSetLayoutCreateInfo layout_ci(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool, binding);
    layout_ci.pNext = &flags_ci;
    m_bindless_descriptor_set_layout = m_device.createDescriptorSetLayoutUnique(layout_ci);
    
    const vk::DescriptorSetAllocateInfo alloc_info(m_descriptor_pool, m_bindless_descriptor_set_layout.get());
    m_bindless_descriptor_set = m_device.allocateDescriptorSets(alloc_info)[0];

    m_free_texture_indices.resize(kMaxBindlessTextures);
    for (uint32_t i = 0; i < kMaxBindlessTextures; ++i)
        m_free_texture_indices[i] = kMaxBindlessTextures - 1 - i;

    vk::DescriptorSetLayoutBinding gradient_binding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment);
    const vk::DescriptorSetLayoutCreateInfo grad_layout_ci({}, gradient_binding);
    m_gradient_descriptor_set_layout = m_device.createDescriptorSetLayoutUnique(grad_layout_ci);
}

void RmlRenderInterface::CreatePipelines(vk::Format colorFormat, vk::Format depthFormat) {
    // --- Push Constants ---
    vk::PushConstantRange push_constant_range(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstants));
    // Main + stencil pipelines: only bindless descriptor set (set 0)
    vk::PipelineLayoutCreateInfo layout_ci({}, m_bindless_descriptor_set_layout.get(), push_constant_range);
    m_pipeline_layout = m_device.createPipelineLayoutUnique(layout_ci);

   // Gradient pipeline: gradient UBO (set 0), bindless textures (set 1)
   std::array layout_sets = {
        m_gradient_descriptor_set_layout.get(), // set 0: gradient UBO
        m_bindless_descriptor_set_layout.get()  // set 1: bindless textures
   };
   vk::PipelineLayoutCreateInfo gradient_layout_ci({}, layout_sets, push_constant_range);
    m_pipeline_gradient_layout = m_device.createPipelineLayoutUnique(gradient_layout_ci);

    // --- Shader Modules ---
    vk::UniqueShaderModule vert_shader = m_device.createShaderModuleUnique({{}, sizeof(shader_vert), reinterpret_cast<const uint32_t*>(shader_vert)});
    vk::UniqueShaderModule frag_shader_gradient = m_device.createShaderModuleUnique({ {}, sizeof(shader_frag_gradient), reinterpret_cast<const uint32_t*>(shader_frag_gradient) });
    vk::UniqueShaderModule frag_shader = m_device.createShaderModuleUnique({{}, sizeof(shader_frag), reinterpret_cast<const uint32_t*>(shader_frag)});

    vk::PipelineShaderStageCreateInfo vert_stage_ci({}, vk::ShaderStageFlagBits::eVertex, vert_shader.get(), "main");
    vk::PipelineShaderStageCreateInfo frag_stage_main_ci({}, vk::ShaderStageFlagBits::eFragment, frag_shader.get(), "main");
    vk::PipelineShaderStageCreateInfo frag_stage_gradient_ci({}, vk::ShaderStageFlagBits::eFragment, frag_shader_gradient.get(), "main");

    // --- Vertex Input ---
    vk::VertexInputBindingDescription binding_desc(0, sizeof(Rml::Vertex), vk::VertexInputRate::eVertex);
    std::array attribute_descs = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Rml::Vertex, position)),
        vk::VertexInputAttributeDescription(1, 0, vk::Format::eR8G8B8A8Unorm, offsetof(Rml::Vertex, colour)),
        vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Rml::Vertex, tex_coord))
    };
    vk::PipelineVertexInputStateCreateInfo vertex_input_ci({}, 1, &binding_desc, static_cast<uint32_t>(attribute_descs.size()), attribute_descs.data());

    // --- Input Assembly, Viewport, Rasterization ---
    vk::PipelineInputAssemblyStateCreateInfo input_assembly_ci({}, vk::PrimitiveTopology::eTriangleList, false);
    vk::PipelineViewportStateCreateInfo viewport_state_ci({}, 1, nullptr, 1, nullptr);
    vk::PipelineRasterizationStateCreateInfo rasterization_ci({}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise, false);
    vk::PipelineMultisampleStateCreateInfo multisample_ci({}, vk::SampleCountFlagBits::e1, false);
    std::array dynamic_states = { vk::DynamicState::eViewport, vk::DynamicState::eScissor, vk::DynamicState::eStencilReference };
    vk::PipelineDynamicStateCreateInfo dynamic_state_ci({}, dynamic_states);
    vk::PipelineRenderingCreateInfo rendering_ci({}, colorFormat, depthFormat, depthFormat);

    // --- Color Blend ---
    vk::PipelineColorBlendAttachmentState color_blend_attachment(
        true,
        vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd,
        vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd,
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    );
    vk::PipelineColorBlendStateCreateInfo color_blend_ci({}, false, vk::LogicOp::eCopy, color_blend_attachment);

    vk::PipelineColorBlendAttachmentState no_color_write_attachment{};
    no_color_write_attachment.colorWriteMask = {};
    vk::PipelineColorBlendStateCreateInfo no_color_blend_ci({}, false, vk::LogicOp::eCopy, no_color_write_attachment);

    // --- Depth/Stencil States ---
    vk::PipelineDepthStencilStateCreateInfo ds_no_stencil({}, false, false, vk::CompareOp::eLessOrEqual);
    
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

    // --- Pipeline Create Info: GETS MUTATED AS WE GO!!
    vk::GraphicsPipelineCreateInfo pipeline_ci{};
    pipeline_ci.pNext = &rendering_ci;
    pipeline_ci.pVertexInputState = &vertex_input_ci;
    pipeline_ci.pInputAssemblyState = &input_assembly_ci;
    pipeline_ci.pViewportState = &viewport_state_ci;
    pipeline_ci.pRasterizationState = &rasterization_ci;
    pipeline_ci.pMultisampleState = &multisample_ci;
    pipeline_ci.pDynamicState = &dynamic_state_ci;
    pipeline_ci.stageCount = 2;

    // --- Main Pipeline ---
    pipeline_ci.layout = m_pipeline_layout.get();
    pipeline_ci.pStages = std::array{ vert_stage_ci, frag_stage_main_ci }.data();
    pipeline_ci.pColorBlendState = &color_blend_ci;
    pipeline_ci.pDepthStencilState = &ds_no_stencil;
    m_pipeline_main = m_device.createGraphicsPipelineUnique({}, pipeline_ci).value;

    // --- Stencil-Only Pipelines (No color output) ---
    // These pipelines can reuse the layout and shaders from the main pipeline.
    pipeline_ci.pColorBlendState = &no_color_blend_ci;

    pipeline_ci.pDepthStencilState = &ds_stencil_gen;
    m_pipeline_stencil_gen = m_device.createGraphicsPipelineUnique({}, pipeline_ci).value;

    pipeline_ci.pDepthStencilState = &ds_stencil_incr;
    m_pipeline_stencil_incr = m_device.createGraphicsPipelineUnique({}, pipeline_ci).value;

    pipeline_ci.pDepthStencilState = &ds_stencil_zero;
    m_pipeline_stencil_zero = m_device.createGraphicsPipelineUnique({}, pipeline_ci).value;

    // --- Stencil Clip Pipeline (with color output) ---
    // This also reuses the main pipeline's layout and shaders.
    pipeline_ci.pColorBlendState = &color_blend_ci;
    pipeline_ci.pDepthStencilState = &ds_stencil_clip;
    m_pipeline_stencil_clip = m_device.createGraphicsPipelineUnique({}, pipeline_ci).value;
    
    // --- Gradient Pipeline ---
    // Set state SPECIFIC to gradient pipelines
    pipeline_ci.layout = m_pipeline_gradient_layout.get();
    pipeline_ci.pStages = std::array{ vert_stage_ci, frag_stage_gradient_ci }.data();
    
    pipeline_ci.pColorBlendState = &color_blend_ci;
    pipeline_ci.pDepthStencilState = &ds_no_stencil;
    m_pipeline_gradient = m_device.createGraphicsPipelineUnique({}, pipeline_ci).value;

    // --- Gradient Stencil Clip Pipeline ---
    // Reuses gradient layout and shaders, just changes the depth/stencil state.
    pipeline_ci.pDepthStencilState = &ds_stencil_clip;
    m_pipeline_gradient_stencil_clip = m_device.createGraphicsPipelineUnique({}, pipeline_ci).value;
}

void RmlRenderInterface::DestroyTexture(const TextureData* texture) {
    if (!texture) return;
    // 1. Return the bindless index to the pool for reuse.
    if (texture->bindless_index != static_cast<uint32_t>(-1))
        m_free_texture_indices.push_back(texture->bindless_index);
    // 2. Destroy the image and its allocation IF we own it.
    if (texture->image)
        vmaDestroyImage(m_allocator, texture->image, texture->allocation);
    // 3. Destroy the sampler IF we own it.
    if (texture->sampler && texture->sampler != m_linear_sampler.get())
        m_device.destroySampler(texture->sampler);
}

void RmlRenderInterface::EnableClipMask(const bool enable) {
    m_clip_mask_enabled = enable;
    if (!enable)
        m_stencil_level = 0;
}

void RmlRenderInterface::RenderToClipMask(Rml::ClipMaskOperation operation, const Rml::CompiledGeometryHandle handle, const Rml::Vector2f translation) {
    auto* geometry = reinterpret_cast<GeometryData*>(handle);

    // Prepare push constants for the mask geometry.
    // The mask itself is never textured.
    PushConstants constants{};
    constants.transform = m_projection_matrix * m_transform_matrix;
    constants.translate = translation;
    constants.texture_id = -1;

    // Bind the descriptor set once. All stencil pipelines use the same layout.
    m_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipeline_layout.get(), 0, m_bindless_descriptor_set, {});

    switch (operation) {
        case Rml::ClipMaskOperation::Set: {
            m_stencil_level = 1;
            m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline_stencil_gen.get());
            m_command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, m_stencil_level);

            // Draw the mask geometry
            m_command_buffer.pushConstants<PushConstants>(m_pipeline_layout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, constants);
            m_command_buffer.bindVertexBuffers(0, geometry->buffer, { geometry->vertex_offset });
            m_command_buffer.bindIndexBuffer(geometry->buffer, geometry->index_offset, vk::IndexType::eUint32);
            m_command_buffer.drawIndexed(geometry->num_indices, 1, 0, 0, 0);
            break;
        }

        case Rml::ClipMaskOperation::SetInverse: {
            m_stencil_level = 1;
            
            // 1. Fill the entire stencil buffer with '1'
            m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline_stencil_gen.get());
            m_command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, m_stencil_level);

            // Draw a fullscreen quad (using an identity matrix for the transform)
            Rml::Vertex quad_vertices[4] = { {{-1.0f, -1.0f}}, {{1.0f, -1.0f}}, {{1.0f, 1.0f}}, {{-1.0f, 1.0f}} };
            int quad_indices[6] = {0, 1, 2, 0, 2, 3};
            if (const auto quad_handle = CompileGeometry({ quad_vertices, 4 }, { quad_indices, 6 })) {
                PushConstants quad_pc = { Rml::Matrix4f::Identity(), {0,0}, -1 };
                auto* quad_geo = reinterpret_cast<GeometryData*>(quad_handle);
                m_command_buffer.pushConstants<PushConstants>(m_pipeline_layout.get(), vk::ShaderStageFlagBits::eVertex, 0, quad_pc);
                m_command_buffer.bindVertexBuffers(0, quad_geo->buffer, { quad_geo->vertex_offset });
                m_command_buffer.bindIndexBuffer(quad_geo->buffer, quad_geo->index_offset, vk::IndexType::eUint32);
                m_command_buffer.drawIndexed(quad_geo->num_indices, 1, 0, 0, 0);
                ReleaseGeometry(quad_handle);
            }

            // 2. Punch a hole (write '0') where the user geometry is
            m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline_stencil_zero.get());
            m_command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, 0);

            // Draw the mask geometry
            m_command_buffer.pushConstants<PushConstants>(m_pipeline_layout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, constants);
            m_command_buffer.bindVertexBuffers(0, geometry->buffer, { geometry->vertex_offset });
            m_command_buffer.bindIndexBuffer(geometry->buffer, geometry->index_offset, vk::IndexType::eUint32);
            m_command_buffer.drawIndexed(geometry->num_indices, 1, 0, 0, 0);
            break;
        }

        case Rml::ClipMaskOperation::Intersect: {
            m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline_stencil_incr.get());
            m_command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, m_stencil_level);
            m_stencil_level++;

            // Draw the mask geometry
            m_command_buffer.pushConstants<PushConstants>(m_pipeline_layout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, constants);
            m_command_buffer.bindVertexBuffers(0, geometry->buffer, { geometry->vertex_offset });
            m_command_buffer.bindIndexBuffer(geometry->buffer, geometry->index_offset, vk::IndexType::eUint32);
            m_command_buffer.drawIndexed(geometry->num_indices, 1, 0, 0, 0);
            break;
        }
    }
}

// SHADER IMPLEMENTATION
Rml::CompiledShaderHandle RmlRenderInterface::CompileShader(const Rml::String& name, const Rml::Dictionary& parameters) {
    auto shader = std::make_unique<CompiledShader>();
    GradientData gradient_data{};
    bool is_gradient = false;

    // Helper to convert RmlUi's 8-bit color to premultiplied float color
    auto ConvertToColorf = [](Rml::ColourbPremultiplied c) -> Rml::Colourf {
        Rml::Colourf result;
        for (int i = 0; i < 4; i++)
            result[i] = static_cast<float>(c[i]) / 255.f;
        return result;
    };

    // Helper to extract the color stop list from RmlUi parameters
    auto ApplyColorStopList = [&](const Rml::Dictionary& shader_parameters) {
        auto it = shader_parameters.find("color_stop_list");
        RMLUI_ASSERT(it != shader_parameters.end());
        const auto& color_stop_list = it->second.GetReference<Rml::ColorStopList>();
        const int num_stops = Rml::Math::Min(static_cast<int>(color_stop_list.size()), 16);

        gradient_data.num_stops = num_stops;
        for (int i = 0; i < num_stops; i++) {
            const auto& stop = color_stop_list[i];
            gradient_data.stops[i].position = stop.position.number;
            gradient_data.stops[i].color = ConvertToColorf(stop.color);
        }
    };
    
    if (name == "linear-gradient" || name == "repeating-linear-gradient") {
        is_gradient = true;
        gradient_data.gradient_function = (name == "repeating-linear-gradient" ? 3 : 0);
        Rml::Vector2f p0 = Rml::Get(parameters, "p0", Rml::Vector2f(0.f));
        Rml::Vector2f p1 = Rml::Get(parameters, "p1", p0 + Rml::Vector2f(1.f, 0.f));
        gradient_data.p = p0;
        gradient_data.v = p1 - p0;
        if (gradient_data.v.SquaredMagnitude() < 1e-6f)
            gradient_data.v = {1.f, 0.f};
    } else if (name == "radial-gradient" || name == "repeating-radial-gradient") {
        is_gradient = true;
        gradient_data.gradient_function = (name == "repeating-radial-gradient" ? 4 : 1);
        gradient_data.p = Rml::Get(parameters, "center", Rml::Vector2f(0.f));
        Rml::Vector2f radius = Rml::Get(parameters, "radius", Rml::Vector2f(1.f));
        gradient_data.v = {1.f / (radius.x * radius.x), 1.f / (radius.y * radius.y)};
    } else if (name == "conic-gradient" || name == "repeating-conic-gradient") {
        is_gradient = true;
        gradient_data.gradient_function = (name == "repeating-conic-gradient" ? 5 : 2);
        gradient_data.p = Rml::Get(parameters, "center", Rml::Vector2f(0.f));
        float angle_rad = Rml::Get(parameters, "angle", 0.0f);
        gradient_data.v = { Rml::Math::Cos(angle_rad), Rml::Math::Sin(angle_rad) };
    }

    if (is_gradient) {
        shader->type = CompiledShaderType::Gradient;
        ApplyColorStopList(parameters);

        // --- Create and populate the gradient UBO ---
        auto ubo = std::make_unique<GradientUBO>(m_allocator, m_device);

        const auto buffer_ci = static_cast<VkBufferCreateInfo>(vk::BufferCreateInfo({}, sizeof(GradientData), vk::BufferUsageFlagBits::eUniformBuffer));
        constexpr VmaAllocationCreateInfo alloc_ci = {VMA_ALLOCATION_CREATE_MAPPED_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU};
        VmaAllocationInfo alloc_info{};
        vmaCreateBuffer(m_allocator, &buffer_ci, &alloc_ci, &ubo->buffer, &ubo->allocation, &alloc_info);

        memcpy(alloc_info.pMappedData, &gradient_data, sizeof(GradientData));

        const vk::DescriptorSetAllocateInfo ds_alloc_info(m_descriptor_pool, m_gradient_descriptor_set_layout.get());
        ubo->descriptor_set = std::move(m_device.allocateDescriptorSetsUnique(ds_alloc_info)[0]);

        vk::DescriptorBufferInfo buf_info(ubo->buffer, 0, sizeof(GradientData));
        vk::WriteDescriptorSet write(ubo->descriptor_set.get(), 0, 0, vk::DescriptorType::eUniformBuffer, {}, buf_info);
        m_device.updateDescriptorSets(write, {});
        
        shader->gradient_ubo = std::move(ubo);
    }

    if (shader->type != CompiledShaderType::Invalid)
        return reinterpret_cast<Rml::CompiledShaderHandle>(shader.release());

    Rml::Log::Message(Rml::Log::LT_WARNING, "Unsupported shader type '%s'.", name.c_str());
    return {};
}

void RmlRenderInterface::RenderShader(Rml::CompiledShaderHandle shader_handle, Rml::CompiledGeometryHandle geometry_handle, Rml::Vector2f translation, Rml::TextureHandle texture) {
    auto* geometry = reinterpret_cast<GeometryData*>(geometry_handle);
    const auto* shader = reinterpret_cast<CompiledShader*>(shader_handle);
    if (!shader || shader->type != CompiledShaderType::Gradient)
        return;

    PushConstants pc{};
    pc.transform = m_projection_matrix * m_transform_matrix;
    pc.translate = translation;
    pc.texture_id = texture ? reinterpret_cast<TextureData*>(texture)->bindless_index : -1;

    // Choose the correct gradient pipeline (with or without stencil clipping)
    const vk::Pipeline pipeline = m_clip_mask_enabled ? m_pipeline_gradient_stencil_clip.get() : m_pipeline_gradient.get();
    m_command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

    // Bind descriptor sets: Set 0 for gradient UBO, Set 1 for bindless textures
    const std::array sets = {
        shader->gradient_ubo->descriptor_set.get(),
        m_bindless_descriptor_set
    };
    m_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipeline_gradient_layout.get(), 0, sets, {});

    m_command_buffer.pushConstants<PushConstants>(m_pipeline_gradient_layout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);

    m_command_buffer.bindVertexBuffers(0, geometry->buffer, { geometry->vertex_offset });
    m_command_buffer.bindIndexBuffer(geometry->buffer, geometry->index_offset, vk::IndexType::eUint32);
    m_command_buffer.drawIndexed(geometry->num_indices, 1, 0, 0, 0);
}

void RmlRenderInterface::ReleaseShader(Rml::CompiledShaderHandle shader_handle) {
    if (shader_handle)
        delete reinterpret_cast<CompiledShader*>(shader_handle);
}