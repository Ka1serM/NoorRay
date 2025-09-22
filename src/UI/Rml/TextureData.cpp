#include "TextureData.h"

TextureData::TextureData(VmaAllocator allocator, vk::Device device, vk::Extent2D extent, vk::Format format, vk::Sampler shared_sampler)
    : m_allocator(allocator), m_device(device)
{
    m_sampler = shared_sampler;
    m_is_sampler_owned = false; // It's a shared sampler.

    // Create an image suitable for rendering to and then sampling from.
    vk::ImageCreateInfo image_ci(
        {},
        vk::ImageType::e2D,
        format,
        vk::Extent3D(extent, 1),
        1, 1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        // CRUCIAL: Must have ColorAttachment and Sampled usage flags. TransferSrc is for the final composite.
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc
    );
    auto vk_image_ci = static_cast<VkImageCreateInfo>(image_ci);

    VmaAllocationCreateInfo alloc_ci = {};
    alloc_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE; // Best for GPU-only resources

    vmaCreateImage(m_allocator, &vk_image_ci, &alloc_ci, &m_image, &m_allocation, nullptr);

    // Create the image view for this render target.
    vk::ImageViewCreateInfo view_ci(
        {},
        m_image,
        vk::ImageViewType::e2D,
        format,
        {}, // component mapping
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} // subresource range
    );
    m_owned_image_view = device.createImageViewUnique(view_ci);
}

// --- Constructor for TextureDatas from raw data ---
TextureData::TextureData(
    VmaAllocator allocator, vk::Device device, vk::Queue graphics_queue, vk::CommandPool command_pool,
    Rml::Span<const Rml::byte> data, Rml::Vector2i dimensions, vk::Sampler shared_sampler
) : m_allocator(allocator), m_device(device)
{
    m_sampler = shared_sampler;
    m_is_sampler_owned = false; // It's a shared sampler we don't own.

    // Staging buffer upload logic
    vk::DeviceSize image_size = data.size();
    vk::Extent3D extent(dimensions.x, dimensions.y, 1);

    auto staging_buffer_ci = static_cast<VkBufferCreateInfo>(vk::BufferCreateInfo({}, image_size, vk::BufferUsageFlagBits::eTransferSrc));
    VmaAllocationCreateInfo staging_alloc_ci = {VMA_ALLOCATION_CREATE_MAPPED_BIT, VMA_MEMORY_USAGE_CPU_ONLY};
    VkBuffer staging_buffer;
    VmaAllocation staging_allocation;
    VmaAllocationInfo staging_alloc_info;
    vmaCreateBuffer(m_allocator, &staging_buffer_ci, &staging_alloc_ci, &staging_buffer, &staging_allocation, &staging_alloc_info);
    memcpy(staging_alloc_info.pMappedData, data.data(), image_size);

    auto image_ci = static_cast<VkImageCreateInfo>(vk::ImageCreateInfo({}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Unorm, extent, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled));
    VmaAllocationCreateInfo image_alloc_ci = {0, VMA_MEMORY_USAGE_GPU_ONLY};
    vmaCreateImage(m_allocator, &image_ci, &image_alloc_ci, &m_image, &m_allocation, nullptr);

    vk::CommandBufferAllocateInfo alloc_info(command_pool, vk::CommandBufferLevel::ePrimary, 1);
    vk::UniqueCommandBuffer cmd = std::move(device.allocateCommandBuffersUnique(alloc_info)[0]);
    cmd->begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    vk::ImageMemoryBarrier to_transfer({}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_image, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    cmd->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, nullptr, nullptr, to_transfer);
    vk::BufferImageCopy copy_region(0, 0, 0, { vk::ImageAspectFlagBits::eColor, 0, 0, 1 }, {}, extent);
    cmd->copyBufferToImage(staging_buffer, m_image, vk::ImageLayout::eTransferDstOptimal, copy_region);
    vk::ImageMemoryBarrier to_shader_read(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_image, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    cmd->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, to_shader_read);
    cmd->end();

    graphics_queue.submit(vk::SubmitInfo({}, {}, *cmd), {});
    graphics_queue.waitIdle();
    vmaDestroyBuffer(m_allocator, staging_buffer, staging_allocation);
    vk::ImageViewCreateInfo view_ci({}, m_image, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Unorm, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    m_owned_image_view = device.createImageViewUnique(view_ci);
}

// --- Constructor for TextureDatas from external views ---
TextureData::TextureData(VmaAllocator allocator, vk::Device device, vk::ImageView external_view)
    : m_allocator(allocator), m_device(device)
{
    m_external_image_view = external_view;
    vk::SamplerCreateInfo sampler_info({}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge);
    m_sampler = m_device.createSampler(sampler_info);
    m_is_sampler_owned = true;
}

// --- Destructor ---
TextureData::~TextureData() {
    if (m_image) {
        vmaDestroyImage(m_allocator, m_image, m_allocation);
    }
    if (m_sampler && m_is_sampler_owned) {
        m_device.destroySampler(m_sampler);
    }
}

// --- Move Semantics ---
TextureData::TextureData(TextureData&& other) noexcept
    : m_allocator(other.m_allocator), m_device(other.m_device), m_image(other.m_image),
      m_allocation(other.m_allocation), m_owned_image_view(std::move(other.m_owned_image_view)),
      m_sampler(other.m_sampler), m_is_sampler_owned(other.m_is_sampler_owned),
      m_external_image_view(other.m_external_image_view), m_bindless_index(other.m_bindless_index)
{
    other.m_image = VK_NULL_HANDLE;
    other.m_allocation = VK_NULL_HANDLE;
    other.m_sampler = VK_NULL_HANDLE;
}
TextureData& TextureData::operator=(TextureData&& other) noexcept {
    if (this != &other) {
        this->~TextureData();
        m_allocator = other.m_allocator; m_device = other.m_device; m_image = other.m_image;
        m_allocation = other.m_allocation; m_owned_image_view = std::move(other.m_owned_image_view);
        m_sampler = other.m_sampler; m_is_sampler_owned = other.m_is_sampler_owned;
        m_external_image_view = other.m_external_image_view; m_bindless_index = other.m_bindless_index;
        other.m_image = VK_NULL_HANDLE; other.m_allocation = VK_NULL_HANDLE; other.m_sampler = VK_NULL_HANDLE;
    }
    return *this;
}