#pragma once

#include "Context.h"
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

class Image {
public:
    Image() = default;
    Image(Context& context, const void* data, int width, int height, vk::Format format);
    Image(Context& context, uint32_t width, uint32_t height, vk::Format format, vk::ImageUsageFlags usage);
    Image(Context& context, uint32_t width, uint32_t height, vk::Format format,
          vk::ImageUsageFlags usage, vk::ExternalMemoryHandleTypeFlagBits externalHandleType);
    void setImageLayout(const vk::CommandBuffer& cmd, vk::ImageLayout newLayout);
    ~Image();

    // Delete copy operations
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    // Enable and declare custom move operations
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    vk::Image getImage() const { return image; }
    vk::ImageView getView() const { return view.get(); }
    vk::DescriptorImageInfo getDescriptorInfo() const { return descImageInfo; }
    vk::Format getFormat() const { return format; }
    vk::DeviceMemory getMemory() const { return externalMemory; }
    vk::DeviceSize getAllocationSize() const { return allocationSize; }
    bool isExportable() const { return externalMemory != VK_NULL_HANDLE; }

    uint32_t getWidth() const { return width; }
    uint32_t getHeight() const { return height; }
private:
    vk::AccessFlags accessFlagsForLayout(vk::ImageLayout layout);
    vk::PipelineStageFlags pipelineStageForAccessFlags(vk::AccessFlags accessFlags);
    vk::Device device;
    VmaAllocator allocator;

    vk::Image image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE; // VMA's handle for the underlying memory
    vk::DeviceMemory externalMemory = VK_NULL_HANDLE;
    vk::DeviceSize allocationSize = 0;

    vk::UniqueImageView view;
    vk::DescriptorImageInfo descImageInfo;

    vk::ImageLayout layout;
    vk::Format format;
    uint32_t width;
    uint32_t height;
};
