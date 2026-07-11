#pragma once

#include "Context.h"
#include <vk_mem_alloc.h>

class Buffer {
public:
    enum class Type {
        Storage,
        Custom
    };

    Buffer();
    ~Buffer();

    Buffer(const Context& context, Type type, vk::DeviceSize size, const void* data = nullptr, vk::BufferUsageFlags usage = {}, vk::MemoryPropertyFlags memoryProps = {});

    // Host-visible, host-coherent buffer allocated outside VMA so its memory can be
    // exported and imported into CUDA (see nr::cuda::UniqueSharedBuffer).
    Buffer(const Context& context, vk::DeviceSize size, vk::BufferUsageFlags usage,
           vk::ExternalMemoryHandleTypeFlagBits externalHandleType);

    // Move Semantics
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    vk::DeviceAddress getDeviceAddress() const { return deviceAddress; }
    const vk::DescriptorBufferInfo& getDescriptorInfo() const { return descBufferInfo; }
    const vk::Buffer& getBuffer() const { return buffer; }
    const vk::DeviceMemory& getMemory() const { return memory; }
    void* getMappedData() const { return mappedData; }
    vk::DeviceSize getSize() const { return descBufferInfo.range; }

private:
    vk::Device device;
    VmaAllocator allocator;
    vk::Buffer buffer;
    VmaAllocation allocation;

    vk::DeviceMemory memory;
    vk::DescriptorBufferInfo descBufferInfo;
    vk::DeviceAddress deviceAddress;
    void* mappedData = nullptr;
};
