#pragma once

#include "Context.h"

class Image {
public:
    Image();
    Image(const Context& context, uint32_t width, uint32_t height, vk::Format format, vk::ImageUsageFlags usage);

    static vk::AccessFlags toAccessFlags(vk::ImageLayout layout);
    static void setImageLayout(vk::CommandBuffer commandBuffer, vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
    static void copyImage(vk::CommandBuffer commandBuffer, vk::Image srcImage, vk::Image dstImage, vk::Extent3D extent);

    vk::UniqueImage image;
    vk::ImageCreateInfo info;
    vk::UniqueImageView view;
    vk::UniqueDeviceMemory memory;
    vk::DescriptorImageInfo descImageInfo;
};