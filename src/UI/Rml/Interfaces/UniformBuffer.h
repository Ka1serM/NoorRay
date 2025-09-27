#pragma once

#include "vk_mem_alloc.h"
#include <utility>

class UniformBuffer : public GpuResource {
public:
    ~UniformBuffer()
    {
        if (m_allocator && m_buffer != VK_NULL_HANDLE && m_allocation != VK_NULL_HANDLE)
            vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
    }

    // The class is non-copyable but movable to allow for ownership transfer.
    UniformBuffer(const UniformBuffer&) = delete;
    UniformBuffer& operator=(const UniformBuffer&) = delete;

    UniformBuffer(UniformBuffer&& other) noexcept
        : m_allocator(other.m_allocator),
          m_device(other.m_device),
          m_buffer(other.m_buffer),
          m_allocation(other.m_allocation),
          m_descriptor_set(std::move(other.m_descriptor_set))
    {
        other.m_buffer = VK_NULL_HANDLE;
        other.m_allocation = VK_NULL_HANDLE;
    }

    UniformBuffer& operator=(UniformBuffer&& other) noexcept
    {
        if (this != &other)
        {
            if (m_allocator && m_buffer != VK_NULL_HANDLE && m_allocation != VK_NULL_HANDLE)
                vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);

            m_allocator = other.m_allocator;
            m_device = other.m_device;
            m_buffer = other.m_buffer;
            m_allocation = other.m_allocation;
            m_descriptor_set = std::move(other.m_descriptor_set);

            other.m_buffer = VK_NULL_HANDLE;
            other.m_allocation = VK_NULL_HANDLE;
        }
        return *this;
    }

    vk::DescriptorSet getDescriptorSet() const
    {
        return m_descriptor_set.get();
    }

    VkBuffer getBuffer() const
    {
        return m_buffer;
    }

    template <typename T>
    static UniformBuffer* Create(
        const T& initial_data,
        VmaAllocator allocator,
        vk::Device device,
        vk::DescriptorPool descriptor_pool,
        vk::DescriptorSetLayout descriptor_set_layout)
    {
        auto* ubo = new UniformBuffer(allocator, device);

        const vk::BufferCreateInfo buffer_ci(
            {},
            sizeof(T),
            vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer);

        constexpr VmaAllocationCreateInfo alloc_ci = {0, VMA_MEMORY_USAGE_CPU_TO_GPU };

        vmaCreateBuffer(allocator,
                        reinterpret_cast<const VkBufferCreateInfo*>(&buffer_ci),
                        &alloc_ci,
                        &ubo->m_buffer,
                        &ubo->m_allocation,
                        nullptr);

        void* mapped_data = nullptr;
        vmaMapMemory(allocator, ubo->m_allocation, &mapped_data);
        memcpy(mapped_data, &initial_data, sizeof(T));
        vmaUnmapMemory(allocator, ubo->m_allocation);

        const vk::DescriptorSetAllocateInfo ds_alloc_info(descriptor_pool, descriptor_set_layout);
        ubo->m_descriptor_set = std::move(device.allocateDescriptorSetsUnique(ds_alloc_info)[0]);

        vk::DescriptorBufferInfo buf_info(ubo->m_buffer, 0, sizeof(T));
        const vk::WriteDescriptorSet write(
            ubo->m_descriptor_set.get(),
            0,
            0,
            vk::DescriptorType::eUniformBuffer,
            {},
            buf_info);

        device.updateDescriptorSets(write, {});

        return ubo;
    }

private:
    UniformBuffer(VmaAllocator allocator, vk::Device device)
        : m_allocator(allocator), m_device(device)
    {}

    VmaAllocator m_allocator = nullptr;
    vk::Device m_device = nullptr;

    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    vk::UniqueDescriptorSet m_descriptor_set;
};
