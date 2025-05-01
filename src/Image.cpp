#include "Image.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

Image Image::fromFile(const Context& context, const std::string& filepath) {

    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(filepath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels)
        throw std::runtime_error("Failed to load texture image!");

    vk::DeviceSize imageSize = texWidth * texHeight * 4;

    // Create staging buffer
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setSize(imageSize);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eTransferSrc);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    vk::UniqueBuffer stagingBuffer = context.device->createBufferUnique(bufferInfo);

    //Allocate memory for staging buffer
    vk::MemoryRequirements memRequirements = context.device->getBufferMemoryRequirements(*stagingBuffer);
    uint32_t memTypeIndex = context.findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.setMemoryTypeIndex(memTypeIndex);
    vk::UniqueDeviceMemory stagingMemory = context.device->allocateMemoryUnique(allocInfo);

    context.device->bindBufferMemory(*stagingBuffer, *stagingMemory, 0);

    //Copy pixels to staging buffer
    void* data = context.device->mapMemory(*stagingMemory, 0, imageSize);
    memcpy(data, pixels, imageSize);
    context.device->unmapMemory(*stagingMemory);

    stbi_image_free(pixels);

    // Create GPU image
    Image image = Image(context, texWidth, texHeight, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);

    //Transition image layout and copy from staging buffer
    context.oneTimeSubmit([&](vk::CommandBuffer cmd) {
        image.setImageLayout(cmd, *image.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

        // Copy buffer to image
        vk::BufferImageCopy copyRegion{};
        copyRegion.setBufferOffset(0);
        copyRegion.setBufferRowLength(0);
        copyRegion.setBufferImageHeight(0);
        copyRegion.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        copyRegion.imageExtent = vk::Extent3D{static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1};

        cmd.copyBufferToImage(*stagingBuffer, *image.image, vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);

        image.setImageLayout(cmd, *image.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        image.descImageInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    });

    return image;
}

Image::Image() = default;

Image::Image(const Context& context, uint32_t width, uint32_t height, vk::Format format, vk::ImageUsageFlags usage) {
    // Create image
    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setExtent({width, height, 1});
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setFormat(format);
    imageInfo.setUsage(usage);
    info = imageInfo;

    image = context.device->createImageUnique(imageInfo);

    // Allocate memory
    vk::MemoryRequirements requirements = context.device->getImageMemoryRequirements(*image);
    uint32_t memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryAllocateInfo memoryInfo;
    memoryInfo.setAllocationSize(requirements.size);
    memoryInfo.setMemoryTypeIndex(memoryTypeIndex);
    memory = context.device->allocateMemoryUnique(memoryInfo);

    // Bind memory and image
    context.device->bindImageMemory(*image, *memory, 0);

    // Create image view
    vk::ImageViewCreateInfo imageViewInfo;
    imageViewInfo.setImage(*image);
    imageViewInfo.setViewType(vk::ImageViewType::e2D);
    imageViewInfo.setFormat(format);
    imageViewInfo.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    view = context.device->createImageViewUnique(imageViewInfo);

    // Set image info
    descImageInfo.setImageView(*view);
    descImageInfo.setImageLayout(vk::ImageLayout::eGeneral);
    context.oneTimeSubmit([&](vk::CommandBuffer commandBuffer) {
        //
        setImageLayout(commandBuffer, *image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    });
}

vk::AccessFlags Image::toAccessFlags(vk::ImageLayout layout) {
    switch (layout) {
        case vk::ImageLayout::eTransferSrcOptimal:
            return vk::AccessFlagBits::eTransferRead;
        case vk::ImageLayout::eTransferDstOptimal:
            return vk::AccessFlagBits::eTransferWrite;
        default:
            return {};
    }
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
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, //
                                  vk::PipelineStageFlagBits::eAllCommands, //
                                  {}, {}, {}, barrier);
}

void Image::copyImage(vk::CommandBuffer commandBuffer, vk::Image srcImage, vk::Image dstImage, vk::Extent3D extent) {
    vk::ImageCopy copyRegion;
    copyRegion.setSrcSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    copyRegion.setDstSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    copyRegion.setExtent(extent);
    commandBuffer.copyImage(srcImage, vk::ImageLayout::eTransferSrcOptimal, dstImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);
}