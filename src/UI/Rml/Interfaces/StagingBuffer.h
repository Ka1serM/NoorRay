#pragma once

#include <Vulkan/Context.h>
#include "vk_mem_alloc.h"
#include <utility> // For std::swap

class StagingBuffer : public GpuResource {
public:
    StagingBuffer(VmaAllocator alloc) : GpuResource(GpuResourceType::StagingBuffer), m_allocator(alloc) {}

    ~StagingBuffer() override
    {
        if (m_buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
    }

    StagingBuffer(const StagingBuffer&) = delete;
    StagingBuffer& operator=(const StagingBuffer&) = delete;
    StagingBuffer(StagingBuffer&& other) noexcept
        : GpuResource(std::move(other)),
          m_allocator(other.m_allocator),
          m_buffer(other.m_buffer),
          m_allocation(other.m_allocation) {
        other.m_buffer = VK_NULL_HANDLE;
        other.m_allocation = VK_NULL_HANDLE;
    }
    StagingBuffer& operator=(StagingBuffer&& other) noexcept {
        swap(other);
        return *this;
    }

    void swap(StagingBuffer& other) noexcept {
        using std::swap;
        swap(m_allocator, other.m_allocator);
        swap(m_buffer, other.m_buffer);
        swap(m_allocation, other.m_allocation);
    }

    VkBuffer& getBuffer() { return m_buffer; }
    VmaAllocation& getAllocation() { return m_allocation; }

private:
    VmaAllocator m_allocator = nullptr;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
};