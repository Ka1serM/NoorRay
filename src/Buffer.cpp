#include "Buffer.h"

Buffer::Buffer() = default;

Buffer::Buffer(const std::shared_ptr<Context>& context, Type type, vk::DeviceSize size, const void* data, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryProps) {

    using Usage = vk::BufferUsageFlagBits;
    using Memory = vk::MemoryPropertyFlagBits;

    if (type != Type::Custom) {
        if (type == Type::AccelInput) {
            usage = Usage::eAccelerationStructureBuildInputReadOnlyKHR | Usage::eStorageBuffer | Usage::eShaderDeviceAddress;
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

    buffer = context->device->createBufferUnique({{}, size, usage});

    const vk::MemoryRequirements requirements = context->device->getBufferMemoryRequirements(*buffer);
    const uint32_t memoryTypeIndex = context->findMemoryType(requirements.memoryTypeBits, memoryProps);

    constexpr vk::MemoryAllocateFlagsInfo flagsInfo{vk::MemoryAllocateFlagBits::eDeviceAddress};
    vk::MemoryAllocateInfo memoryInfo;
    memoryInfo.setAllocationSize(requirements.size);
    memoryInfo.setMemoryTypeIndex(memoryTypeIndex);
    memoryInfo.setPNext(&flagsInfo);

    memory = context->device->allocateMemoryUnique(memoryInfo);
    context->device->bindBufferMemory(*buffer, *memory, 0);

    const vk::BufferDeviceAddressInfoKHR bufferDeviceAI{*buffer};
    deviceAddress = context->device->getBufferAddressKHR(&bufferDeviceAI);

    descBufferInfo.setBuffer(*buffer);
    descBufferInfo.setOffset(0);
    descBufferInfo.setRange(size);

    if (data) {
        void* mapped = context->device->mapMemory(*memory, 0, size);
        memcpy(mapped, data, size);
        context->device->unmapMemory(*memory);
    }
}
