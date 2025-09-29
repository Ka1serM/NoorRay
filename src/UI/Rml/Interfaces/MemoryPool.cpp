#include "MemoryPool.h"

MemoryPool::MemoryPool(VmaAllocator allocator, vk::DeviceSize total_size, vk::BufferUsageFlags usage)
    : m_allocator(allocator) 
{
    const vk::BufferCreateInfo buffer_ci({}, total_size, usage);
    
    VmaAllocationCreateInfo alloc_ci = {};
    alloc_ci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    alloc_ci.usage = VMA_MEMORY_USAGE_CPU_TO_GPU; 

    VkBuffer vk_buffer;
    vmaCreateBuffer(m_allocator, 
                    reinterpret_cast<const VkBufferCreateInfo*>(&buffer_ci), 
                    &alloc_ci, 
                    &vk_buffer, 
                    &m_allocation, 
                    nullptr);
    m_buffer = vk_buffer;

    VmaVirtualBlockCreateInfo block_ci = {};
    block_ci.size = total_size;
    vmaCreateVirtualBlock(&block_ci, &m_virtual_block);

    // Persistently map the buffer
    vmaMapMemory(m_allocator, m_allocation, reinterpret_cast<void**>(&m_mapped_data_base));
}

MemoryPool::~MemoryPool() {
    if (m_allocator) {
        vmaUnmapMemory(m_allocator, m_allocation);
        vmaDestroyVirtualBlock(m_virtual_block);
        vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
    }
}

std::optional<SubAllocation> MemoryPool::allocate(vk::DeviceSize size) {
    VmaVirtualAllocationCreateInfo alloc_info = {};
    alloc_info.size = size;

    VmaVirtualAllocation vma_alloc;
    vk::DeviceSize offset;

    if (vmaVirtualAllocate(m_virtual_block, &alloc_info, &vma_alloc, &offset) != VK_SUCCESS)
        return std::nullopt; // Allocation failed

    SubAllocation result;
    result.allocation = vma_alloc;
    result.offset = offset;
    result.size = size;
    result.mapped_data = m_mapped_data_base + offset;

    return result;
}

void MemoryPool::free(const SubAllocation& sub_allocation) {
    if (sub_allocation.allocation)
        vmaVirtualFree(m_virtual_block, sub_allocation.allocation);
}