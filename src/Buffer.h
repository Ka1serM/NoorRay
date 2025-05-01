#pragma once

#include "Context.h"

class Buffer {
public:
    enum class Type {
        Storage,
        Scratch,
        AccelInput,
        AccelStorage,
        ShaderBindingTable,
    };

    Buffer();
    Buffer(const Context& context, Type type, vk::DeviceSize size, const void* data = nullptr);

    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::DescriptorBufferInfo descBufferInfo;
    uint64_t deviceAddress = 0;
};