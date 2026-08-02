#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

class Context;

namespace nr::cuda
{

// One VMA allocation with distinct CPU, CUDA, and Vulkan views. CUDA external
// memory does not preserve the CPU virtual address, so callers must use the
// appropriate accessor for the processor that consumes the data.
class UniqueSharedBuffer
{
public:
    UniqueSharedBuffer() = default;
    ~UniqueSharedBuffer() noexcept { reset(); }

    UniqueSharedBuffer(const UniqueSharedBuffer&) = delete;
    UniqueSharedBuffer& operator=(const UniqueSharedBuffer&) = delete;
    UniqueSharedBuffer(UniqueSharedBuffer&& other) noexcept;
    UniqueSharedBuffer& operator=(UniqueSharedBuffer&& other) noexcept;

    void create(Context& context, vk::DeviceSize bytes, vk::BufferUsageFlags usage);
    void reset() noexcept;

    void* hostPointer() const { return hostPointer_; }
    void* cudaPointer() const { return cudaPointer_; }
    vk::Buffer vulkanBuffer() const { return buffer_; }
    const vk::DescriptorBufferInfo& descriptorInfo() const { return descriptorInfo_; }
    vk::DeviceSize size() const { return descriptorInfo_.range; }

private:
    VmaAllocator allocator_{};
    VmaPool pool_{};
    VkBuffer buffer_{};
    VmaAllocation allocation_{};
    void* hostPointer_{};
    cudaExternalMemory_t cudaMemory_{};
    void* cudaPointer_{};
    vk::DescriptorBufferInfo descriptorInfo_{};
};

}
