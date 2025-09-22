#pragma once

#include <vulkan/vulkan.hpp>
#include "vk_mem_alloc.h"
#include <memory>

class UniformBuffer
{
public:
    template <typename T>
    static std::unique_ptr<UniformBuffer> Create(
        const T& initial_data,
        VmaAllocator allocator,
        vk::Device device,
        vk::DescriptorPool descriptor_pool,
        vk::DescriptorSetLayout descriptor_set_layout);

    /**
     * @brief Destroys the Vulkan buffer and frees the associated memory.
     */
    ~UniformBuffer();

    // The class is non-copyable but movable to allow for ownership transfer.
    UniformBuffer(const UniformBuffer&) = delete;
    UniformBuffer& operator=(const UniformBuffer&) = delete;
    UniformBuffer(UniformBuffer&& other) noexcept;
    UniformBuffer& operator=(UniformBuffer&& other) noexcept;

    /**
     * @brief Updates the buffer's content with new data.
     * @tparam T The type of the data structure.
     * @param data The new data to copy to the buffer.
     */
    template <typename T>
    void Update(const T& data);

    /**
     * @brief Gets the Vulkan descriptor set for this uniform buffer.
     * @return The raw vk::DescriptorSet handle.
     */
    vk::DescriptorSet getDescriptorSet() const;

private:
    UniformBuffer(VmaAllocator allocator, vk::Device device);

    VmaAllocator m_allocator = nullptr;
    vk::Device m_device = nullptr;

    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    vk::UniqueDescriptorSet m_descriptor_set;

    // A persistent pointer to the mapped buffer memory for efficient updates.
    void* m_mapped_data = nullptr;
};

template <typename T>
std::unique_ptr<UniformBuffer> UniformBuffer::Create(
    const T& initial_data,
    VmaAllocator allocator,
    vk::Device device,
    vk::DescriptorPool descriptor_pool,
    vk::DescriptorSetLayout descriptor_set_layout)
{
    // Use `new` because the constructor is private. `std::make_unique` cannot access it.
    auto ubo = std::unique_ptr<UniformBuffer>(new UniformBuffer(allocator, device));

    const auto buffer_ci = static_cast<VkBufferCreateInfo>(vk::BufferCreateInfo(
        {}, sizeof(T), vk::BufferUsageFlagBits::eUniformBuffer));

    // VMA_ALLOCATION_CREATE_MAPPED_BIT ensures that alloc_info.pMappedData is valid
    // for the lifetime of the allocation.
    constexpr VmaAllocationCreateInfo alloc_ci = {
        VMA_ALLOCATION_CREATE_MAPPED_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU // Ideal for buffers frequently updated by the CPU.
    };

    VmaAllocationInfo alloc_info{};
    vmaCreateBuffer(allocator, &buffer_ci, &alloc_ci, &ubo->m_buffer, &ubo->m_allocation, &alloc_info);

    // Store the persistent pointer to the mapped memory.
    ubo->m_mapped_data = alloc_info.pMappedData;

    // Use the Update method to perform the initial data copy.
    ubo->Update(initial_data);

    // Allocate and configure the descriptor set.
    const vk::DescriptorSetAllocateInfo ds_alloc_info(descriptor_pool, descriptor_set_layout);
    ubo->m_descriptor_set = std::move(device.allocateDescriptorSetsUnique(ds_alloc_info)[0]);

    vk::DescriptorBufferInfo buf_info(ubo->m_buffer, 0, sizeof(T));
    const vk::WriteDescriptorSet write(ubo->m_descriptor_set.get(), 0, 0, vk::DescriptorType::eUniformBuffer, {}, buf_info);
    device.updateDescriptorSets(write, {});

    return ubo;
}

template <typename T>
void UniformBuffer::Update(const T& data)
{
    // It's good practice to ensure the mapped pointer is valid before using it.
    if (m_mapped_data)
        memcpy(m_mapped_data, &data, sizeof(T));
}