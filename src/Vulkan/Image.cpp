#include <stb_image.h>
#include <memory>
#include "Image.h"

Image::Image(Context& context, const void* data, int width, int height, vk::Format format)
{
    currentLayout = vk::ImageLayout::eUndefined;

    size_t pixelSize = 0;

    switch (format) {
        case vk::Format::eR8Unorm:
        case vk::Format::eR8Srgb:
            pixelSize = 1; break;
        case vk::Format::eR8G8Unorm:
        case vk::Format::eR8G8Srgb:
            pixelSize = 2; break;
        case vk::Format::eR8G8B8Unorm:
        case vk::Format::eR8G8B8Srgb:
            pixelSize = 3; break;
        case vk::Format::eB8G8R8A8Unorm:
        case vk::Format::eR8G8B8A8Unorm:
        case vk::Format::eR8G8B8A8Srgb:
            pixelSize = 4; break;
        case vk::Format::eR16Sfloat:
            pixelSize = 2; break;
        case vk::Format::eR16G16Sfloat:
            pixelSize = 4; break;
        case vk::Format::eR16G16B16A16Sfloat:
            pixelSize = 8; break;
        case vk::Format::eR32Sfloat:
            pixelSize = 4; break;
        case vk::Format::eR32G32Sfloat:
            pixelSize = 8; break;
        case vk::Format::eR32G32B32A32Sfloat:
            pixelSize = 16; break;
        default:
            throw std::runtime_error("Unsupported vk::Format in Image constructor");
    }

    vk::DeviceSize imageSize = width * height * pixelSize;

    // Staging buffer
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setSize(imageSize);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eTransferSrc);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    vk::UniqueBuffer stagingBuffer = context.device->createBufferUnique(bufferInfo);

    vk::MemoryRequirements memRequirements = context.device->getBufferMemoryRequirements(*stagingBuffer);
    uint32_t memTypeIndex = context.findMemoryType(memRequirements.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.setMemoryTypeIndex(memTypeIndex);
    vk::UniqueDeviceMemory stagingMemory = context.device->allocateMemoryUnique(allocInfo);
    context.device->bindBufferMemory(*stagingBuffer, *stagingMemory, 0);

    // Upload data
    void* mapped = context.device->mapMemory(*stagingMemory, 0, imageSize);
    memcpy(mapped, data, imageSize);
    context.device->unmapMemory(*stagingMemory);

    // Create image
    vk::ImageCreateInfo imageInfo{};
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setExtent({ static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 });
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setFormat(format);
    imageInfo.setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
    info = imageInfo;

    image = context.device->createImageUnique(imageInfo);

    vk::MemoryRequirements imgReqs = context.device->getImageMemoryRequirements(*image);
    uint32_t imgMemType = context.findMemoryType(imgReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryAllocateInfo imgAlloc{};
    imgAlloc.setAllocationSize(imgReqs.size);
    imgAlloc.setMemoryTypeIndex(imgMemType);
    memory = context.device->allocateMemoryUnique(imgAlloc);
    context.device->bindImageMemory(*image, *memory, 0);

    // Image view
    vk::ImageViewCreateInfo viewInfo{};
    viewInfo.setImage(*image);
    viewInfo.setViewType(vk::ImageViewType::e2D);
    viewInfo.setFormat(format);
    viewInfo.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    view = context.device->createImageViewUnique(viewInfo);

    // Copy & transition
    context.oneTimeSubmit([&](vk::CommandBuffer cmd) {
        setImageLayout(cmd, image.get(), vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

        vk::BufferImageCopy region{};
        region.setBufferOffset(0);
        region.setBufferRowLength(0);
        region.setBufferImageHeight(0);
        region.setImageSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
        region.setImageExtent({ static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 });

        cmd.copyBufferToImage(*stagingBuffer, *image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

        setImageLayout(cmd, image.get(), vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    });

    descImageInfo.setImageView(*view);
    descImageInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
}

Image::Image(Context& context, const void* rgbaData, int texWidth, int texHeight)
{
    currentLayout = vk::ImageLayout::eUndefined;

    vk::DeviceSize imageSize = texWidth * texHeight * 4; // 4 = RGBA8

    // Create staging buffer
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setSize(imageSize);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eTransferSrc);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    vk::UniqueBuffer stagingBuffer = context.device->createBufferUnique(bufferInfo);

    vk::MemoryRequirements memRequirements = context.device->getBufferMemoryRequirements(*stagingBuffer);
    uint32_t memTypeIndex = context.findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.setMemoryTypeIndex(memTypeIndex);
    vk::UniqueDeviceMemory stagingMemory = context.device->allocateMemoryUnique(allocInfo);

    context.device->bindBufferMemory(*stagingBuffer, *stagingMemory, 0);

    void* mappedData = context.device->mapMemory(*stagingMemory, 0, imageSize);
    memcpy(mappedData, rgbaData, imageSize);
    context.device->unmapMemory(*stagingMemory);

    // Create GPU image
    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setExtent({ static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1 });
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setFormat(vk::Format::eR8G8B8A8Unorm);
    imageInfo.setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
    info = imageInfo;

    image = context.device->createImageUnique(imageInfo);

    vk::MemoryRequirements requirements = context.device->getImageMemoryRequirements(*image);
    uint32_t memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryAllocateInfo memoryInfo {};
    memoryInfo.setAllocationSize(requirements.size);
    memoryInfo.setMemoryTypeIndex(memoryTypeIndex);
    memory = context.device->allocateMemoryUnique(memoryInfo);

    context.device->bindImageMemory(*image, *memory, 0);

    // Create image view
    vk::ImageViewCreateInfo imageViewInfo;
    imageViewInfo.setImage(*image);
    imageViewInfo.setViewType(vk::ImageViewType::e2D);
    imageViewInfo.setFormat(vk::Format::eR8G8B8A8Unorm);
    imageViewInfo.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    view = context.device->createImageViewUnique(imageViewInfo);

    // Copy buffer to image and transition layout
    context.oneTimeSubmit([&](vk::CommandBuffer commandBuffer) {
        setImageLayout(commandBuffer, image.get(), vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

        vk::BufferImageCopy copyRegion{};
        copyRegion.setBufferOffset(0);
        copyRegion.setBufferRowLength(0);
        copyRegion.setBufferImageHeight(0);
        copyRegion.setImageSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
        copyRegion.setImageExtent({ static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1 });

        commandBuffer.copyBufferToImage(*stagingBuffer, *image, vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);

        setImageLayout(commandBuffer, image.get(), vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    });

    // Final descriptor
    descImageInfo.setImageView(*view);
    descImageInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
}

Image::Image(Context& context, uint32_t width, uint32_t height, vk::Format format, vk::ImageUsageFlags usage) {
    // Create image
    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setExtent({width, height, 1});
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setFormat(format);
    imageInfo.setUsage(usage);
    imageInfo.setInitialLayout(vk::ImageLayout::eUndefined); // Default
    currentLayout = vk::ImageLayout::eUndefined;

    info = imageInfo;
    image = context.device->createImageUnique(imageInfo);

    // Allocate memory
    vk::MemoryRequirements requirements = context.device->getImageMemoryRequirements(*image);
    uint32_t memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryAllocateInfo memoryInfo {};
    memoryInfo.setAllocationSize(requirements.size);
    memoryInfo.setMemoryTypeIndex(memoryTypeIndex);
    memory = context.device->allocateMemoryUnique(memoryInfo);

    // Bind memory
    context.device->bindImageMemory(*image, *memory, 0);

    // Create image view
    vk::ImageViewCreateInfo imageViewInfo;
    imageViewInfo.setImage(*image);
    imageViewInfo.setViewType(vk::ImageViewType::e2D);
    imageViewInfo.setFormat(format);
    imageViewInfo.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    view = context.device->createImageViewUnique(imageViewInfo);

    // Update descriptor image info
    descImageInfo.setImageView(*view);
    descImageInfo.setImageLayout(vk::ImageLayout::eGeneral);
    descImageInfo.setSampler(nullptr); // if needed

    // Transition image layout
    context.oneTimeSubmit([&](vk::CommandBuffer commandBuffer) {
        setImageLayout(commandBuffer,image.get(), vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    });
}

vk::AccessFlags Image::toAccessFlags(vk::ImageLayout layout) {
    switch (layout) {
        case vk::ImageLayout::eUndefined:
            return {};
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
            return {};
    }
}

void Image::setImageLayout(const vk::CommandBuffer& commandBuffer, vk::ImageLayout newLayout) {
    setImageLayout(commandBuffer, image.get(), currentLayout, newLayout);
}

void Image::setImageLayout(const vk::CommandBuffer& commandBuffer, const vk::Image& image, vk::ImageLayout oldLayout, const vk::ImageLayout newLayout) {
    vk::ImageMemoryBarrier barrier;
    barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setImage(image);
    barrier.setOldLayout(oldLayout);
    barrier.setNewLayout(newLayout);
    barrier.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    barrier.setSrcAccessMask(toAccessFlags(oldLayout));
    barrier.setDstAccessMask(toAccessFlags(newLayout));
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,{}, {}, {}, barrier);
}