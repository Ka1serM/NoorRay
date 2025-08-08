#pragma once

#include "Context.h"
#include <vulkan/vulkan.hpp>

class Image {
private:
    vk::DescriptorImageInfo descImageInfo;
    vk::ImageLayout currentLayout;

    vk::UniqueImageView view;
    vk::UniqueImage image;
    vk::UniqueDeviceMemory memory;
    vk::ImageCreateInfo info;

public:
    Image(Context& context, const void* floatData, int texWidth, int texHeight, vk::Format format);
    Image(Context& context, const void* rgbaData, int texWidth, int texHeight);
    Image(Context& context, uint32_t width, uint32_t height, vk::Format format, vk::ImageUsageFlags usage);

    void setImageLayout(const vk::CommandBuffer& commandBuffer, vk::ImageLayout newLayout);
    static void setImageLayout(const vk::CommandBuffer& commandBuffer, const vk::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
    void update(const Context& context, const void* data, size_t dataSize);

    static vk::AccessFlags toAccessFlags(vk::ImageLayout layout);

    const vk::DescriptorImageInfo& getDescriptorImageInfo() const { return descImageInfo; }
    vk::ImageLayout getCurrentLayout() const { return currentLayout; }
    const vk::ImageView& getImageView() const { return view.get(); }
    const vk::Image& getImage() const { return image.get(); }
    const vk::ImageCreateInfo& getImageCreateInfo() const { return info; }
    vk::Format getFormat() const { return info.format; }
};
