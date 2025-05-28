#include <stb_image.h>
#include <memory>

#include "Image.h"

Image::Image(Context& context, const std::string& filepath) {
    currentLayout = vk::ImageLayout::eUndefined;

    int texWidth, texHeight, texChannels;

    //Load image and force 4 channels (RGBA)
    stbi_uc* rawPixels = stbi_load(filepath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!rawPixels)
        throw std::runtime_error("Failed to load texture image!");

    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(rawPixels, stbi_image_free);

    //if (texChannels != 4)
    //    std::cerr << "Warning: Loaded image has " << texChannels << " channels, forcing to 4.\n";

    vk::DeviceSize imageSize = texWidth * texHeight * 4;

    //Create staging buffer
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

    //Map and copy image data
    void* data = context.device->mapMemory(*stagingMemory, 0, imageSize);
    memcpy(data, pixels.get(), imageSize);
    context.device->unmapMemory(*stagingMemory);

    //Create GPU image
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
    vk::MemoryAllocateInfo memoryInfo;
    memoryInfo.setAllocationSize(requirements.size);
    memoryInfo.setMemoryTypeIndex(memoryTypeIndex);
    memory = context.device->allocateMemoryUnique(memoryInfo);

    context.device->bindImageMemory(*image, *memory, 0);

    //Create image view
    vk::ImageViewCreateInfo imageViewInfo;
    imageViewInfo.setImage(*image);
    imageViewInfo.setViewType(vk::ImageViewType::e2D);
    imageViewInfo.setFormat(vk::Format::eR8G8B8A8Unorm);
    imageViewInfo.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    view = context.device->createImageViewUnique(imageViewInfo);

    //Copy and transition
    context.oneTimeSubmit([&](vk::CommandBuffer commandBuffer) {

        setImageLayout(commandBuffer, image.get(), vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

        vk::BufferImageCopy copyRegion{};
        copyRegion.setBufferOffset(0);
        copyRegion.setBufferRowLength(0);
        copyRegion.setBufferImageHeight(0);
        copyRegion.imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
        copyRegion.imageExtent = vk::Extent3D{
            static_cast<uint32_t>(texWidth),
            static_cast<uint32_t>(texHeight),
            1
        };

        commandBuffer.copyBufferToImage(*stagingBuffer, *image, vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
        Image::setImageLayout(commandBuffer, image.get(), vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
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
    vk::MemoryAllocateInfo memoryInfo;
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

void Image::setImageLayout(vk::CommandBuffer commandBuffer, vk::ImageLayout newLayout) {
    setImageLayout(commandBuffer, image.get(), currentLayout, newLayout);
}

void Image::setImageLayout(vk::CommandBuffer commandBuffer, vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
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