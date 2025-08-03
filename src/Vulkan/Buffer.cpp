#include "Buffer.h"

Buffer::Buffer(const Context& context, const Type type, vk::DeviceSize size, const void* data, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryProps) {

    using Usage = vk::BufferUsageFlagBits;
    using Memory = vk::MemoryPropertyFlagBits;

    if (type != Type::Custom) {
        if (type == Type::AccelInput) {
            usage = Usage::eStorageBuffer | Usage::eShaderDeviceAddress;
            if (context.isRtxSupported())
                usage |= Usage::eAccelerationStructureBuildInputReadOnlyKHR;
            memoryProps = Memory::eHostVisible | Memory::eHostCoherent;
        } else if (type == Type::Scratch) {
            usage = Usage::eStorageBuffer | Usage::eShaderDeviceAddress;
            memoryProps = Memory::eDeviceLocal;
        } else if (type == Type::AccelStorage) {
            usage = Usage::eAccelerationStructureStorageKHR | Usage::eShaderDeviceAddress;
            memoryProps = Memory::eDeviceLocal;
        } else if (type == Type::ShaderBindingTable) {
            usage = Usage::eShaderBindingTableKHR | Usage::eShaderDeviceAddress;
            memoryProps = Memory::eHostVisible | Memory::eHostCoherent;
        } else if (type == Type::Storage) {
            usage = Usage::eStorageBuffer | Usage::eShaderDeviceAddress;
            memoryProps = Memory::eHostVisible | Memory::eHostCoherent | Memory::eDeviceLocal;
        }
    }

    buffer = context.getDevice().createBufferUnique({{}, size, usage});

    const vk::MemoryRequirements requirements = context.getDevice().getBufferMemoryRequirements(*buffer);
    const uint32_t memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits, memoryProps);

    constexpr vk::MemoryAllocateFlagsInfo flagsInfo{vk::MemoryAllocateFlagBits::eDeviceAddress};
    vk::MemoryAllocateInfo memoryInfo;
    memoryInfo.setAllocationSize(requirements.size);
    memoryInfo.setMemoryTypeIndex(memoryTypeIndex);
    memoryInfo.setPNext(&flagsInfo);

    memory = context.getDevice().allocateMemoryUnique(memoryInfo);
    context.getDevice().bindBufferMemory(*buffer, *memory, 0);

    const vk::BufferDeviceAddressInfoKHR bufferDeviceAI{*buffer};
    deviceAddress = context.getDevice().getBufferAddressKHR(&bufferDeviceAI);

    descBufferInfo.setBuffer(*buffer);
    descBufferInfo.setOffset(0);
    descBufferInfo.setRange(size);

    if (data) {
        void* mapped = context.getDevice().mapMemory(*memory, 0, size);
        memcpy(mapped, data, size);
        context.getDevice().unmapMemory(*memory);
    }
}
