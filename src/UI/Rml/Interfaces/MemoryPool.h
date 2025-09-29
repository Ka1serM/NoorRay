#pragma once

#include <vulkan/vulkan.hpp>
#include "vk_mem_alloc.h"
#include <optional>

struct SubAllocation {
    VmaVirtualAllocation allocation = nullptr;
    vk::DeviceSize offset = 0;
    vk::DeviceSize size = 0;
    void* mapped_data = nullptr;
};

class MemoryPool {
public:
    MemoryPool(VmaAllocator allocator, vk::DeviceSize total_size, vk::BufferUsageFlags usage);
    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = delete;
    MemoryPool& operator=(MemoryPool&&) = delete;

    std::optional<SubAllocation> allocate(vk::DeviceSize size);

    void free(const SubAllocation& sub_allocation);

    vk::Buffer getBuffer() const { return m_buffer; }

private:
    VmaAllocator m_allocator = nullptr;
    VmaVirtualBlock m_virtual_block = nullptr;

    vk::Buffer m_buffer = nullptr;
    VmaAllocation m_allocation = nullptr;
    
    uint8_t* m_mapped_data_base = nullptr; // Base pointer to the start of the mapped buffer
};