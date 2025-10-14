#include "Image.h"
#include <stdexcept>
#include <cstring>
#include <iostream>

// Move Constructor Implementation
Image::Image(Image&& other) noexcept
    : device(other.device),
      allocator(other.allocator),
      image(other.image),
      allocation(other.allocation),
      view(std::move(other.view)), // vk::Unique handles have their own move
      descImageInfo(other.descImageInfo),
      layout(other.layout),
        format(other.format),
      width(other.width),
      height(other.height)
{
    // Reset the source object so its destructor does nothing
    other.image = VK_NULL_HANDLE;
    other.allocation = VK_NULL_HANDLE;
    other.device = VK_NULL_HANDLE;
    other.allocator = VK_NULL_HANDLE;
}

// Move Assignment Operator Implementation
Image& Image::operator=(Image&& other) noexcept
{
    if (this != &other) {
        // Clean up existing resources first
        if (image && allocation)
            vmaDestroyImage(allocator, image, allocation);
        view.reset();

        // Pilfer the resources from the other object
        device = other.device;
        allocator = other.allocator;
        image = other.image;
        allocation = other.allocation;
        view = std::move(other.view);
        descImageInfo = other.descImageInfo;
        layout = other.layout;
        format = other.format;
        width = other.width;
        height = other.height;

        // Reset the source object
        other.image = VK_NULL_HANDLE;
        other.allocation = VK_NULL_HANDLE;
        other.device = VK_NULL_HANDLE;
        other.allocator = VK_NULL_HANDLE;
    }
    return *this;
}

Image::Image(Context& context, const void* data, int texWidth, int texHeight, vk::Format format)
    : device(context.getDevice()),
      allocator(context.getAllocator()),
      layout(vk::ImageLayout::eUndefined),
        format(format),
      width(texWidth),
      height(texHeight)
{
    if (width <= 0 || height <= 0 || !data)
        throw std::runtime_error("Image: Invalid dimensions or null data pointer provided.");

    // Determine pixel size to calculate total image size
    size_t pixelSize;
    switch (format) {
        case vk::Format::eR8G8B8A8Unorm:
        case vk::Format::eR8G8B8A8Srgb:
        case vk::Format::eB8G8R8A8Unorm:
        case vk::Format::eR32Sfloat:          pixelSize = 4; break;
        case vk::Format::eR32G32Sfloat:       pixelSize = 8; break;
        case vk::Format::eR32G32B32A32Sfloat: pixelSize = 16; break;
        default:
            throw std::runtime_error("Image: Unsupported vk::Format provided for data upload.");
    }
    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(width) * height * pixelSize;

    // Create Staging Buffer
    vk::BufferCreateInfo stagingBufferInfo;
    stagingBufferInfo.setSize(imageSize);
    stagingBufferInfo.setUsage(vk::BufferUsageFlagBits::eTransferSrc);

    VmaAllocationCreateInfo stagingAllocInfo = {};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY; // We want a buffer on the CPU to copy to.

    vk::Buffer stagingBuffer;
    VmaAllocation stagingAllocation;
    vmaCreateBuffer(allocator, reinterpret_cast<const VkBufferCreateInfo*>(&stagingBufferInfo), &stagingAllocInfo, reinterpret_cast<VkBuffer*>(&stagingBuffer), &stagingAllocation, nullptr);

    // Map memory, copy data, and unmap
    void* mappedData;
    vmaMapMemory(allocator, stagingAllocation, &mappedData);
    memcpy(mappedData, data, imageSize);
    vmaUnmapMemory(allocator, stagingAllocation);

    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D)
             .setExtent({ width, height, 1 })
             .setMipLevels(1)
             .setArrayLayers(1)
             .setFormat(format)
             .setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)
             .setInitialLayout(vk::ImageLayout::eUndefined);

    VmaAllocationCreateInfo imgAllocInfo = {};
    imgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY; // We want this image to live on the GPU for fast sampling.

    vmaCreateImage(allocator, reinterpret_cast<const VkImageCreateInfo*>(&imageInfo), &imgAllocInfo, reinterpret_cast<VkImage*>(&this->image), &this->allocation, nullptr);

    // 4. Copy data from staging buffer to final image
    context.oneTimeSubmit([&](const vk::CommandBuffer cmd) {
        setImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);

        vk::BufferImageCopy region;
        region.setImageSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
        region.setImageExtent({ width, height, 1 });
        cmd.copyBufferToImage(stagingBuffer, this->image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

        setImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    });

    // 5. Clean up staging resources
    vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);

    // 6. Create Image View
    vk::ImageViewCreateInfo viewInfo;
    viewInfo.setImage(this->image)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(format)
            .setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    this->view = device.createImageViewUnique(viewInfo);

    descImageInfo.setImageView(*view);
    descImageInfo.setImageLayout(this->layout);
}

Image::Image(Context& context, uint32_t w, uint32_t h, vk::Format format, vk::ImageUsageFlags usage)
    : device(context.getDevice()),
      allocator(context.getAllocator()),
      layout(vk::ImageLayout::eUndefined),
    format(format),
      width(w),
      height(h)
{
    // Create Final Image with VMA (No staging buffer needed)
    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D)
             .setExtent({ width, height, 1 })
             .setMipLevels(1)
             .setArrayLayers(1)
             .setFormat(format)
             .setUsage(usage)
             .setInitialLayout(vk::ImageLayout::eUndefined);

    VmaAllocationCreateInfo imgAllocInfo = {};
    imgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(allocator, reinterpret_cast<const VkImageCreateInfo*>(&imageInfo), &imgAllocInfo, reinterpret_cast<VkImage*>(&this->image), &this->allocation, nullptr);

    // Create Image View
    vk::ImageViewCreateInfo viewInfo;
    viewInfo.setImage(this->image)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(format)
            .setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    this->view = device.createImageViewUnique(viewInfo);
    
    descImageInfo.setImageView(*view);
    descImageInfo.setImageLayout(this->layout);
}

Image::~Image() {
    if (image != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE)
        vmaDestroyImage(allocator, image, allocation);
}

void Image::setImageLayout(const vk::CommandBuffer& cmd, const vk::ImageLayout newLayout) {
    if (layout == newLayout)
        return;

    vk::ImageMemoryBarrier barrier{};
    barrier.image = image;
    barrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    barrier.oldLayout = layout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    barrier.srcAccessMask = accessFlagsForLayout(layout);
    barrier.dstAccessMask = accessFlagsForLayout(newLayout);

    const vk::PipelineStageFlags srcStage = pipelineStageForAccessFlags(barrier.srcAccessMask);
    const vk::PipelineStageFlags dstStage = pipelineStageForAccessFlags(barrier.dstAccessMask);
    
    cmd.pipelineBarrier(srcStage, dstStage, {}, nullptr, nullptr, barrier);

    this->layout = newLayout;
}

vk::AccessFlags Image::accessFlagsForLayout(vk::ImageLayout layout) {
    switch (layout) {
    case vk::ImageLayout::eUndefined:
        return vk::AccessFlags{};
    case vk::ImageLayout::eGeneral:
        return vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
    case vk::ImageLayout::eColorAttachmentOptimal:
        return vk::AccessFlagBits::eColorAttachmentWrite;
    case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        return vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    case vk::ImageLayout::eShaderReadOnlyOptimal:
        return vk::AccessFlagBits::eShaderRead;
    case vk::ImageLayout::eTransferSrcOptimal:
        return vk::AccessFlagBits::eTransferRead;
    case vk::ImageLayout::eTransferDstOptimal:
        return vk::AccessFlagBits::eTransferWrite;
    case vk::ImageLayout::ePresentSrcKHR:
        return vk::AccessFlagBits::eMemoryRead;
    default:
        throw std::runtime_error("Unsupported layout in accessFlagsForLayout!");
    }
}

vk::PipelineStageFlags Image::pipelineStageForAccessFlags(vk::AccessFlags accessFlags) {
    if (accessFlags == vk::AccessFlags{})
        // No access needed means no corresponding pipeline stage, so we can start at the top.
        return vk::PipelineStageFlagBits::eTopOfPipe;

    // This covers all transfer operations
    if ((accessFlags & (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)) != vk::AccessFlags{})
        return vk::PipelineStageFlagBits::eTransfer;
    // This covers all shader read/write operations
    if ((accessFlags & (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite)) != vk::AccessFlags{})
        // We can be conservative here and sync with all shader stages.
        return vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader;
    if ((accessFlags & vk::AccessFlagBits::eColorAttachmentWrite) != vk::AccessFlags{})
        return vk::PipelineStageFlagBits::eColorAttachmentOutput;
    if ((accessFlags & vk::AccessFlagBits::eDepthStencilAttachmentWrite) != vk::AccessFlags{})
        return vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests;
    if ((accessFlags & vk::AccessFlagBits::eMemoryRead) != vk::AccessFlags{}) {
        return vk::PipelineStageFlagBits::eHost; // Typically for host reads like present
    }
    
    std::cerr << "[WARNING] Unsupported access flags encountered in pipelineStageForAccessFlags. "
              << "Using a conservative 'eAllCommands' pipeline stage. Consider adding a specific case for better performance."
              << std::endl;
              
    return vk::PipelineStageFlagBits::eAllCommands;
}