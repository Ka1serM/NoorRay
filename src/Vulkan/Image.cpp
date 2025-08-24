#include "Image.h"
#include <stdexcept>
#include <iostream>
#include <string>
#include <cstring>

Image::Image(Context& context, const void* data, int width, int height, vk::Format format)
{
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Image constructor (floatData): Invalid dimensions (W=" + std::to_string(width) + ", H=" + std::to_string(height) + ")");
    }

    currentLayout = vk::ImageLayout::eUndefined;

    size_t pixelSize = 0;
    switch (format) {
        case vk::Format::eR8Unorm:      case vk::Format::eR8Srgb:           pixelSize = 1; break;
        case vk::Format::eR8G8Unorm:    case vk::Format::eR8G8Srgb:         pixelSize = 2; break;
        case vk::Format::eR8G8B8Unorm:  case vk::Format::eR8G8B8Srgb:       pixelSize = 3; break;
        case vk::Format::eB8G8R8A8Unorm:
        case vk::Format::eR8G8B8A8Unorm:
        case vk::Format::eR8G8B8A8Srgb:                                     pixelSize = 4; break;
        case vk::Format::eR16Sfloat:                                        pixelSize = 2; break;
        case vk::Format::eR16G16Sfloat:                                     pixelSize = 4; break;
        case vk::Format::eR16G16B16A16Sfloat:                               pixelSize = 8; break;
        case vk::Format::eR32Sfloat:                                        pixelSize = 4; break;
        case vk::Format::eR32G32Sfloat:                                     pixelSize = 8; break;
        case vk::Format::eR32G32B32A32Sfloat:                               pixelSize = 16; break;
        default:
            throw std::runtime_error("Unsupported vk::Format in Image constructor (floatData): " + std::to_string(static_cast<int>(format)));
    }

    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(width) * height * pixelSize;

    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setSize(imageSize);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eTransferSrc);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    vk::UniqueBuffer stagingBuffer = context.getDevice().createBufferUnique(bufferInfo);

    vk::MemoryRequirements memRequirements = context.getDevice().getBufferMemoryRequirements(*stagingBuffer);
    uint32_t memTypeIndex = context.findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.setMemoryTypeIndex(memTypeIndex);
    vk::UniqueDeviceMemory stagingMemory = context.getDevice().allocateMemoryUnique(allocInfo);
    context.getDevice().bindBufferMemory(*stagingBuffer, *stagingMemory, 0);

    void* mapped = context.getDevice().mapMemory(*stagingMemory, 0, imageSize);
    if (data && imageSize > 0) {
        std::memcpy(mapped, data, imageSize);
    } else if (!data && imageSize > 0) {
        context.getDevice().unmapMemory(*stagingMemory);
        throw std::runtime_error("Image constructor (floatData): 'data' pointer is null but imageSize is " + std::to_string(imageSize));
    }
    context.getDevice().unmapMemory(*stagingMemory);

    vk::ImageCreateInfo imageInfo{};
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setExtent({ static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 });
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setFormat(format);
    imageInfo.setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
    imageInfo.setInitialLayout(vk::ImageLayout::eUndefined);
    info = imageInfo;

    image = context.getDevice().createImageUnique(imageInfo);

    vk::MemoryRequirements imgReqs = context.getDevice().getImageMemoryRequirements(*image);
    uint32_t imgMemType = context.findMemoryType(imgReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryAllocateInfo imgAlloc{};
    imgAlloc.setAllocationSize(imgReqs.size);
    imgAlloc.setMemoryTypeIndex(imgMemType);
    memory = context.getDevice().allocateMemoryUnique(imgAlloc);
    context.getDevice().bindImageMemory(*image, *memory, 0);

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo.setImage(*image);
    viewInfo.setViewType(vk::ImageViewType::e2D);
    viewInfo.setFormat(format);
    viewInfo.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    view = context.getDevice().createImageViewUnique(viewInfo);

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
    std::cout << "Image (floatData) created for W=" << width << ", H=" << height << ", Format=" << static_cast<int>(format) << std::endl;
}

Image::Image(Context& context, const void* rgbaData, int texWidth, int texHeight)
{
    if (texWidth <= 0 || texHeight <= 0)
        throw std::runtime_error("Image constructor (rgbaData): Invalid dimensions (W=" + std::to_string(texWidth) + ", H=" + std::to_string(texHeight) + ")");

    currentLayout = vk::ImageLayout::eUndefined;

    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(texWidth) * texHeight * 4;

    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setSize(imageSize);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eTransferSrc);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    vk::UniqueBuffer stagingBuffer = context.getDevice().createBufferUnique(bufferInfo);

    vk::MemoryRequirements memRequirements = context.getDevice().getBufferMemoryRequirements(*stagingBuffer);
    uint32_t memTypeIndex = context.findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.setMemoryTypeIndex(memTypeIndex);
    vk::UniqueDeviceMemory stagingMemory = context.getDevice().allocateMemoryUnique(allocInfo);
    context.getDevice().bindBufferMemory(*stagingBuffer, *stagingMemory, 0);

    void* mappedData = context.getDevice().mapMemory(*stagingMemory, 0, imageSize);
    if (rgbaData && imageSize > 0) {
        std::memcpy(mappedData, rgbaData, imageSize);
    } else if (!rgbaData && imageSize > 0) {
        context.getDevice().unmapMemory(*stagingMemory);
        throw std::runtime_error("Image constructor (rgbaData): 'rgbaData' pointer is null but imageSize is " + std::to_string(imageSize));
    }
    context.getDevice().unmapMemory(*stagingMemory);

    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setExtent({ static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1 });
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setFormat(vk::Format::eR8G8B8A8Unorm);
    imageInfo.setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
    imageInfo.setInitialLayout(vk::ImageLayout::eUndefined);
    info = imageInfo;

    image = context.getDevice().createImageUnique(imageInfo);

    vk::MemoryRequirements requirements = context.getDevice().getImageMemoryRequirements(*image);
    uint32_t memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryAllocateInfo memoryInfo {};
    memoryInfo.setAllocationSize(requirements.size);
    memoryInfo.setMemoryTypeIndex(memoryTypeIndex);
    memory = context.getDevice().allocateMemoryUnique(memoryInfo);
    context.getDevice().bindImageMemory(*image, *memory, 0);

    vk::ImageViewCreateInfo imageViewInfo;
    imageViewInfo.setImage(*image);
    imageViewInfo.setViewType(vk::ImageViewType::e2D);
    imageViewInfo.setFormat(vk::Format::eR8G8B8A8Unorm);
    imageViewInfo.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    view = context.getDevice().createImageViewUnique(imageViewInfo);

    context.oneTimeSubmit([&](const vk::CommandBuffer commandBuffer) {
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

    descImageInfo.setImageView(*view);
    descImageInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    std::cout << "Image (rgbaData) created for W=" << texWidth << ", H=" << texHeight << ", Format=R8G8B8A8Unorm" << std::endl;
}

Image::Image(Context& context, uint32_t width, uint32_t height, vk::Format format, vk::ImageUsageFlags usage) {
    if (width == 0 || height == 0)
        throw std::runtime_error("Image constructor (blank): Invalid dimensions (W=" + std::to_string(width) + ", H=" + std::to_string(height) + ")");

    currentLayout = vk::ImageLayout::eUndefined;

    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setExtent({width, height, 1});
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setFormat(format);
    imageInfo.setUsage(usage);
    imageInfo.setInitialLayout(vk::ImageLayout::eUndefined);
    info = imageInfo;

    image = context.getDevice().createImageUnique(imageInfo);

    vk::MemoryRequirements requirements = context.getDevice().getImageMemoryRequirements(*image);
    uint32_t memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryAllocateInfo memoryInfo {};
    memoryInfo.setAllocationSize(requirements.size);
    memoryInfo.setMemoryTypeIndex(memoryTypeIndex);
    memory = context.getDevice().allocateMemoryUnique(memoryInfo);
    context.getDevice().bindImageMemory(*image, *memory, 0);

    vk::ImageViewCreateInfo imageViewInfo;
    imageViewInfo.setImage(*image);
    imageViewInfo.setViewType(vk::ImageViewType::e2D);
    imageViewInfo.setFormat(format);
    imageViewInfo.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    view = context.getDevice().createImageViewUnique(imageViewInfo);

    descImageInfo.setImageView(*view);
    descImageInfo.setImageLayout(vk::ImageLayout::eGeneral);
    descImageInfo.setSampler(nullptr);

    context.oneTimeSubmit([&](const vk::CommandBuffer commandBuffer) {
        setImageLayout(commandBuffer, image.get(), vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    });
    currentLayout = vk::ImageLayout::eGeneral;
    std::cout << " Image (blank) created for W=" << width << ", H=" << height << ", Format=" << static_cast<int>(format) << ", Usage=" << static_cast<uint32_t>(usage) << std::endl;
}

vk::AccessFlags Image::toAccessFlags(const vk::ImageLayout layout) {
    switch (layout) {
        case vk::ImageLayout::eUndefined:               return {};
        case vk::ImageLayout::eGeneral:                 return vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
        case vk::ImageLayout::eColorAttachmentOptimal:  return vk::AccessFlagBits::eColorAttachmentWrite;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal: return vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        case vk::ImageLayout::eShaderReadOnlyOptimal:   return vk::AccessFlagBits::eShaderRead;
        case vk::ImageLayout::eTransferSrcOptimal:      return vk::AccessFlagBits::eTransferRead;
        case vk::ImageLayout::eTransferDstOptimal:      return vk::AccessFlagBits::eTransferWrite;
        case vk::ImageLayout::ePresentSrcKHR:           return vk::AccessFlagBits::eMemoryRead;
        default:                                        return {};
    }
}

void Image::setImageLayout(const vk::CommandBuffer& commandBuffer, vk::ImageLayout newLayout) {
    setImageLayout(commandBuffer, image.get(), currentLayout, newLayout);
    currentLayout = newLayout;
}

void Image::setImageLayout(const vk::CommandBuffer& commandBuffer, const vk::Image& img, vk::ImageLayout oldLayout, const vk::ImageLayout newLayout) {
    vk::ImageMemoryBarrier barrier;
    barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setImage(img);
    barrier.setOldLayout(oldLayout);
    barrier.setNewLayout(newLayout);
    barrier.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    barrier.setSrcAccessMask(toAccessFlags(oldLayout));
    barrier.setDstAccessMask(toAccessFlags(newLayout));

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,{}, 0, nullptr, 0, nullptr, 1, &barrier);
}

void Image::update(Context& context, const void* data, size_t dataSize) {
    if (!data || dataSize == 0) {
        std::cerr << "Warning: Attempted to update image with null data or zero size." << std::endl;
        return;
    }

    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setSize(dataSize);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eTransferSrc);
    vk::UniqueBuffer stagingBuffer = context.getDevice().createBufferUnique(bufferInfo);

    vk::MemoryRequirements memRequirements = context.getDevice().getBufferMemoryRequirements(*stagingBuffer);
    uint32_t memTypeIndex = context.findMemoryType(memRequirements.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.setMemoryTypeIndex(memTypeIndex);
    vk::UniqueDeviceMemory stagingMemory = context.getDevice().allocateMemoryUnique(allocInfo);
    context.getDevice().bindBufferMemory(*stagingBuffer, *stagingMemory, 0);

    void* mapped = context.getDevice().mapMemory(*stagingMemory, 0, dataSize);
    std::memcpy(mapped, data, dataSize);
    context.getDevice().unmapMemory(*stagingMemory);

    context.oneTimeSubmit([&](const vk::CommandBuffer cmd) {
        setImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
        
        vk::BufferImageCopy region{};
        region.setBufferOffset(0);
        region.setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
        region.setImageOffset({0, 0, 0});
        region.setImageExtent({info.extent.width, info.extent.height, 1});
        
        cmd.copyBufferToImage(*stagingBuffer, *image, vk::ImageLayout::eTransferDstOptimal, region);
        setImageLayout(cmd, vk::ImageLayout::eGeneral);
    });
}
