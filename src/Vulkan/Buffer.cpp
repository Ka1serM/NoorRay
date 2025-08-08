#include "Buffer.h"

Buffer::Buffer() {
    descBufferInfo.setBuffer(VK_NULL_HANDLE);
    descBufferInfo.setOffset(0);
    descBufferInfo.setRange(VK_WHOLE_SIZE);
}

Buffer::Buffer(const Context& context, const Type type, vk::DeviceSize size, const void* data, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryProps)
{
    // Auto configure usage and memory flags for known types
    if (type != Type::Custom) {
        switch (type) {
            case Type::AccelInput:
                usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
                if (context.isRtxSupported())
                    usage |= vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
                memoryProps = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
                break;

            case Type::Scratch:
                usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
                memoryProps = vk::MemoryPropertyFlagBits::eDeviceLocal;
                break;

            case Type::AccelStorage:
                usage = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
                memoryProps = vk::MemoryPropertyFlagBits::eDeviceLocal;
                break;

            case Type::ShaderBindingTable:
                usage = vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
                memoryProps = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
                break;

            case Type::Storage:
                usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
                memoryProps = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
                break;

            default:
                break;
        }
    }

    // Create buffer
    buffer = context.getDevice().createBufferUnique({{}, size, usage});

    const vk::MemoryRequirements requirements = context.getDevice().getBufferMemoryRequirements(*buffer);
    const uint32_t memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits, memoryProps);

    vk::MemoryAllocateInfo memoryInfo{};
    memoryInfo.setAllocationSize(requirements.size);
    memoryInfo.setMemoryTypeIndex(memoryTypeIndex);

    vk::MemoryAllocateFlagsInfo flagsInfo{};
    if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
        flagsInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
        memoryInfo.setPNext(&flagsInfo);
    }

    // Allocate and bind memory
    memory = context.getDevice().allocateMemoryUnique(memoryInfo);
    context.getDevice().bindBufferMemory(*buffer, *memory, 0);

    // Retrieve device address only if needed
    if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
        vk::BufferDeviceAddressInfo addressInfo{*buffer};
        deviceAddress = context.getDevice().getBufferAddressKHR(&addressInfo);
    }

    // Descriptor buffer info
    descBufferInfo.setBuffer(*buffer);
    descBufferInfo.setOffset(0);
    descBufferInfo.setRange(size);

    // Optional initial data upload
    if (data) {
        void* mapped = context.getDevice().mapMemory(*memory, 0, size);
        std::memcpy(mapped, data, size);

        if (!(memoryProps & vk::MemoryPropertyFlagBits::eHostCoherent)) {
            vk::MappedMemoryRange range{};
            range.memory = *memory;
            range.offset = 0;
            range.size = size;
            context.getDevice().flushMappedMemoryRanges(range);
        }

        context.getDevice().unmapMemory(*memory);
    }
}