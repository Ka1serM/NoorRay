#pragma once

#include <RmlUi/Core/Types.h>
#include <vulkan/vulkan.hpp>
#include "vk_mem_alloc.h"

class TextureData {
public:
    // --- CONSTRUCTOR for TextureDatas from file/memory data ---
    TextureData(VmaAllocator allocator, vk::Device device, vk::Queue graphics_queue, vk::CommandPool command_pool, Rml::Span<const Rml::byte> data, Rml::Vector2i dimensions, vk::Sampler shared_sampler);
    // --- CONSTRUCTOR for TextureDatas from external views (read-only) ---
    TextureData(VmaAllocator allocator, vk::Device device, vk::ImageView external_view);
    // --- NEW: CONSTRUCTOR for render target TextureDatas ---
    TextureData(VmaAllocator allocator, vk::Device device, vk::Extent2D extent, vk::Format format, vk::Sampler shared_sampler);

    ~TextureData();

    // Rule of 5: Movable, not copyable
    TextureData(const TextureData&) = delete;
    TextureData& operator=(const TextureData&) = delete;
    TextureData(TextureData&& other) noexcept;
    TextureData& operator=(TextureData&& other) noexcept;

    VkImage getImage() const { return m_image; }
    vk::ImageView getImageView() const {return m_owned_image_view ? m_owned_image_view.get() : m_external_image_view;}
    vk::Sampler getSampler() const { return m_sampler; }
    uint32_t getBindlessIndex() const {return m_bindless_index;}
    void setBindlessIndex(uint32_t index) {m_bindless_index = index;}

private:
    VmaAllocator m_allocator = nullptr;
    vk::Device m_device = nullptr;
    VkImage m_image = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    vk::UniqueImageView m_owned_image_view;
    vk::Sampler m_sampler = VK_NULL_HANDLE;
    bool m_is_sampler_owned = false;

    // For externally-managed views
    vk::ImageView m_external_image_view = VK_NULL_HANDLE;

    // For bindless rendering
    uint32_t m_bindless_index = -1;
};