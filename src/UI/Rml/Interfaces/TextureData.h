#pragma once

#include <RmlUi/Core/Types.h>
#include "vk_mem_alloc.h"
#include <utility>
#include "StagingBuffer.h"

class TextureData : public GpuResource {
public:
    static std::pair<TextureData*, StagingBuffer*>
    Create(
        VmaAllocator allocator,
        vk::Device device,
        vk::CommandBuffer cmd,
        Rml::Span<const Rml::byte> data,
        Rml::Vector2i dimensions,
        vk::Sampler shared_sampler);

    // Constructor for render targets
    TextureData(VmaAllocator allocator, vk::Device device, vk::Extent2D extent, vk::Format format, vk::Sampler shared_sampler);
    // Constructor for external views
    TextureData(VmaAllocator allocator, vk::Device device, vk::ImageView external_view);
    
    ~TextureData() override;

    TextureData(const TextureData&) = delete;
    TextureData& operator=(const TextureData&) = delete;
    TextureData(TextureData&& other) noexcept;
    TextureData& operator=(TextureData&& other) noexcept;

    vk::ImageView getImageView() const { return m_owned_image_view ? m_owned_image_view.get() : m_external_image_view; }
    vk::Sampler getSampler() const { return m_sampler; }
    uint32_t getBindlessIndex() const { return m_bindless_index; }
    void setBindlessIndex(const uint32_t index) { m_bindless_index = index; }

    void swap(TextureData& other) noexcept;

private:
    TextureData(VmaAllocator allocator, vk::Device device);

    VmaAllocator m_allocator = nullptr;
    vk::Device m_device = nullptr;

    VkImage m_image = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    vk::UniqueImageView m_owned_image_view;

    vk::Sampler m_sampler = VK_NULL_HANDLE;
    bool m_is_sampler_owned = false;

    vk::ImageView m_external_image_view = VK_NULL_HANDLE;
    uint32_t m_bindless_index = static_cast<uint32_t>(-1);
};


inline std::pair<TextureData*, StagingBuffer*>
TextureData::Create(
    VmaAllocator allocator,
    vk::Device device,
    vk::CommandBuffer cmd,
    Rml::Span<const Rml::byte> data,
    Rml::Vector2i dimensions,
    vk::Sampler shared_sampler)
{
    vk::DeviceSize image_size = data.size();
    if (image_size == 0) {
        return { nullptr, nullptr };
    }

    // Create the objects that will be returned
    auto texture_ptr = new TextureData(allocator, device);
    auto staging_buffer_ptr = new StagingBuffer(allocator);

    texture_ptr->m_sampler = shared_sampler;
    texture_ptr->m_is_sampler_owned = false;
    
    vk::Extent3D extent(dimensions.x, dimensions.y, 1);

    // 1. Create and fill the staging buffer
    auto staging_buffer_ci = static_cast<VkBufferCreateInfo>(vk::BufferCreateInfo({}, image_size, vk::BufferUsageFlagBits::eTransferSrc));
    VmaAllocationCreateInfo staging_alloc_ci = {VMA_ALLOCATION_CREATE_MAPPED_BIT, VMA_MEMORY_USAGE_CPU_ONLY};
    VmaAllocationInfo staging_alloc_info;
    vmaCreateBuffer(allocator, &staging_buffer_ci, &staging_alloc_ci, &staging_buffer_ptr->getBuffer(), &staging_buffer_ptr->getAllocation(), &staging_alloc_info);
    memcpy(staging_alloc_info.pMappedData, data.data(), image_size);

    // 2. Create the final GPU image
    auto image_ci = static_cast<VkImageCreateInfo>(vk::ImageCreateInfo({}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Unorm, extent, 1, 1,
        vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled));
    VmaAllocationCreateInfo image_alloc_ci = {0, VMA_MEMORY_USAGE_GPU_ONLY};
    vmaCreateImage(allocator, &image_ci, &image_alloc_ci, &texture_ptr->m_image, &texture_ptr->m_allocation, nullptr);

    // 3. Record commands to transition, copy, and transition the image layout
    vk::ImageMemoryBarrier to_transfer({}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, texture_ptr->m_image, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, nullptr, nullptr, to_transfer);
    
    vk::BufferImageCopy copy_region(0, 0, 0, { vk::ImageAspectFlagBits::eColor, 0, 0, 1 }, {}, extent);
    vk::Buffer staging_buffer_handle = staging_buffer_ptr->getBuffer();
    cmd.copyBufferToImage(staging_buffer_handle, texture_ptr->m_image, vk::ImageLayout::eTransferDstOptimal, copy_region);
    
    vk::ImageMemoryBarrier to_shader_read(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, texture_ptr->m_image, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, to_shader_read);
    
    // 4. Create the image view
    vk::ImageViewCreateInfo view_ci({}, texture_ptr->m_image, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Unorm, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    texture_ptr->m_owned_image_view = device.createImageViewUnique(view_ci);

    // Return ownership of both new objects to the caller
    return { std::move(texture_ptr), std::move(staging_buffer_ptr) };
}

inline TextureData::TextureData(VmaAllocator allocator, vk::Device device)
    : GpuResource(GpuResourceType::Texture) // <-- The type is correctly set here
    , m_allocator(allocator), m_device(device)
{}

inline TextureData::TextureData(VmaAllocator allocator, vk::Device device, vk::Extent2D extent, vk::Format format, vk::Sampler shared_sampler)
    : GpuResource(GpuResourceType::Texture) // <-- The type is correctly set here
   , m_allocator(allocator), m_device(device), m_sampler(shared_sampler)
{
    vk::ImageCreateInfo image_ci({}, vk::ImageType::e2D, format, vk::Extent3D(extent, 1), 1, 1,
        vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc);
    auto vk_image_ci = static_cast<VkImageCreateInfo>(image_ci);
    VmaAllocationCreateInfo alloc_ci = { 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
    vmaCreateImage(m_allocator, &vk_image_ci, &alloc_ci, &m_image, &m_allocation, nullptr);
    vk::ImageViewCreateInfo view_ci({}, m_image, vk::ImageViewType::e2D, format, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    m_owned_image_view = device.createImageViewUnique(view_ci);
}

inline TextureData::TextureData(VmaAllocator allocator, vk::Device device, vk::ImageView external_view)
    : GpuResource(GpuResourceType::Texture) // <-- The type is correctly set here
    , m_allocator(allocator), m_device(device), m_external_image_view(external_view)
{
    vk::SamplerCreateInfo sampler_info({}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge);
    m_sampler = device.createSampler(sampler_info);
    m_is_sampler_owned = true;
}

inline TextureData::~TextureData() {
    if (m_image != VK_NULL_HANDLE)
        vmaDestroyImage(m_allocator, m_image, m_allocation);
    if (m_sampler != VK_NULL_HANDLE && m_is_sampler_owned)
        m_device.destroySampler(m_sampler);
}
    
inline TextureData::TextureData(TextureData&& other) noexcept: GpuResource(GpuResourceType::Texture) // <-- The type is correctly set here
{
    swap(other);
}

inline TextureData& TextureData::operator=(TextureData&& other) noexcept {
    swap(other);
    return *this;
}

inline void TextureData::swap(TextureData& other) noexcept {
    using std::swap;
    swap(m_allocator, other.m_allocator);
    swap(m_device, other.m_device);
    swap(m_image, other.m_image);
    swap(m_allocation, other.m_allocation);
    swap(m_owned_image_view, other.m_owned_image_view);
    swap(m_sampler, other.m_sampler);
    swap(m_is_sampler_owned, other.m_is_sampler_owned);
    swap(m_external_image_view, other.m_external_image_view);
    swap(m_bindless_index, other.m_bindless_index);
}