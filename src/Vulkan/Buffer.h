#pragma once

#include <memory>

#include "Context.h"

class Buffer {
public:
    enum class Type {
        AccelInput,
        Scratch,
        AccelStorage,
        ShaderBindingTable,
        Storage,
        Custom
    };
    Buffer() = default;
    Buffer(Context& context, Type type, vk::DeviceSize size, const void* data = nullptr, vk::BufferUsageFlags usage = {}, vk::MemoryPropertyFlags memoryProps = {});

    //Rule of Five
    ~Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept = default;
    Buffer& operator=(Buffer&& other) noexcept = default;

    vk::DeviceAddress getDeviceAddress() const { return deviceAddress; }
    const vk::DescriptorBufferInfo& getDescriptorInfo() const { return descBufferInfo; }
    const vk::Buffer& getBuffer() const { return buffer.get(); }

private:
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::DescriptorBufferInfo descBufferInfo{};
    vk::DeviceAddress deviceAddress{};
};