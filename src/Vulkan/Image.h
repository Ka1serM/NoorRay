#pragma once

#include <memory>
#include "Context.h"

class Image {
public:

    Image(Context& context, const std::string &filepath);

    Image(Context& context, uint32_t width, uint32_t height, vk::Format format, vk::ImageUsageFlags usage);

    void setImageLayout(vk::CommandBuffer commandBuffer, vk::ImageLayout newLayout);

    static void setImageLayout(vk::CommandBuffer commandBuffer, vk::Image image, vk::ImageLayout oldLayout,
                               vk::ImageLayout newLayout);

    static vk::AccessFlags toAccessFlags(vk::ImageLayout layout);

    vk::UniqueImage image;
    vk::ImageCreateInfo info;
    vk::UniqueImageView view;
    vk::UniqueDeviceMemory memory;
    vk::DescriptorImageInfo descImageInfo;
    vk::ImageLayout currentLayout;
};