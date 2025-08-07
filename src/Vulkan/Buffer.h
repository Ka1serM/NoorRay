#pragma once

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

    Buffer();
    Buffer(const Context& context, Type type, vk::DeviceSize size, const void* data = nullptr, vk::BufferUsageFlags usage = {}, vk::MemoryPropertyFlags memoryProps = {});

    vk::DeviceAddress getDeviceAddress() const { return deviceAddress; }
    const vk::DescriptorBufferInfo& getDescriptorInfo() const { return descBufferInfo; }
    const vk::Buffer& getBuffer() const { return buffer.get(); }
    const vk::DeviceMemory & getMemory() const { return memory.get(); }

private:
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::DescriptorBufferInfo descBufferInfo{};
    vk::DeviceAddress deviceAddress{};
};