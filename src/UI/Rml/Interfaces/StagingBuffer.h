#pragma once

#include <Vulkan/Context.h>
#include "vk_mem_alloc.h"

class StagingBuffer : public GpuResource {
public:
    StagingBuffer(VmaAllocator alloc) : allocator(alloc), buffer(VK_NULL_HANDLE), allocation(VK_NULL_HANDLE) {}

    ~StagingBuffer() {
        if (buffer != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE)
            vmaDestroyBuffer(allocator, buffer, allocation);
    }

    // Rule of 5: Make it non-copyable but movable
    StagingBuffer(const StagingBuffer&) = delete;
    StagingBuffer& operator=(const StagingBuffer&) = delete;
    StagingBuffer(StagingBuffer&& other) noexcept 
        : allocator(other.allocator), buffer(other.buffer), allocation(other.allocation) {
        other.buffer = VK_NULL_HANDLE;
        other.allocation = VK_NULL_HANDLE;
    }
    StagingBuffer& operator=(StagingBuffer&& other) noexcept {
        return *this;
    }
    
    VmaAllocator allocator;
    VkBuffer buffer;
    VmaAllocation allocation;
};