#pragma once

#include "Context.h"

class Image {
public:
    Image(Context& context, const void* floatData, int texWidth, int texHeight, vk::Format format);
    Image(Context& context, const void* rgbaData, int texWidth, int texHeight);
    Image(Context& context, uint32_t width, uint32_t height, vk::Format format, vk::ImageUsageFlags usage);

    void setImageLayout(const vk::CommandBuffer& commandBuffer, vk::ImageLayout newLayout);

    static void setImageLayout(const vk::CommandBuffer& commandBuffer, const vk::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);

    static vk::AccessFlags toAccessFlags(vk::ImageLayout layout);

    vk::DescriptorImageInfo descImageInfo;
    vk::ImageLayout currentLayout;

    vk::UniqueImageView view;
    vk::UniqueImage image;
    vk::UniqueDeviceMemory memory;
    vk::ImageCreateInfo info;
};