#include "UniformBuffer.h"
#include <utility> // For std::move

UniformBuffer::UniformBuffer(VmaAllocator allocator, vk::Device device)
    : m_allocator(allocator), m_device(device)
{
}

UniformBuffer::~UniformBuffer()
{
    if (m_allocator && m_buffer != VK_NULL_HANDLE && m_allocation != VK_NULL_HANDLE)
        vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
}

// --- Move Semantics ---

UniformBuffer::UniformBuffer(UniformBuffer&& other) noexcept
    : m_allocator(other.m_allocator),
      m_device(other.m_device),
      m_buffer(other.m_buffer),
      m_allocation(other.m_allocation),
      m_descriptor_set(std::move(other.m_descriptor_set)),
      m_mapped_data(other.m_mapped_data)
{
    // Invalidate the 'other' object to prevent it from freeing the resources
    // when its destructor is called.
    other.m_buffer = VK_NULL_HANDLE;
    other.m_allocation = VK_NULL_HANDLE;
    other.m_mapped_data = nullptr;
}

UniformBuffer& UniformBuffer::operator=(UniformBuffer&& other) noexcept
{
    if (this != &other)
    {
        // Clean up any existing resources this object owns before overwriting.
        if (m_allocator && m_buffer != VK_NULL_HANDLE && m_allocation != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
        }

        // Move the resources from the 'other' object to this one.
        m_allocator = other.m_allocator;
        m_device = other.m_device;
        m_buffer = other.m_buffer;
        m_allocation = other.m_allocation;
        m_descriptor_set = std::move(other.m_descriptor_set);
        m_mapped_data = other.m_mapped_data;

        // Invalidate the 'other' object.
        other.m_buffer = VK_NULL_HANDLE;
        other.m_allocation = VK_NULL_HANDLE;
        other.m_mapped_data = nullptr;
    }
    return *this;
}

vk::DescriptorSet UniformBuffer::getDescriptorSet() const
{
    return m_descriptor_set.get();
}
