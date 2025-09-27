#include "Buffer.h"
#include <stdexcept>

Buffer::Buffer() {
    // Default constructor
    descBufferInfo.setBuffer(VK_NULL_HANDLE);
    descBufferInfo.setOffset(0);
    descBufferInfo.setRange(VK_WHOLE_SIZE);
}

Buffer::~Buffer() {
    // Let VMA handle the destruction of the buffer and its memory
    if (allocator && buffer && allocation)
        vmaDestroyBuffer(allocator, buffer, allocation);
}

// --- Move Semantics Implementation ---
Buffer::Buffer(Buffer&& other) noexcept
    : allocator(other.allocator),
      buffer(other.buffer),
      allocation(other.allocation),
      memory(other.memory),
      descBufferInfo(other.descBufferInfo),
      deviceAddress(other.deviceAddress)
{
    // Nullify the other object so its destructor does nothing
    other.allocator = VK_NULL_HANDLE;
    other.buffer = VK_NULL_HANDLE;
    other.allocation = VK_NULL_HANDLE;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        // Destroy existing resources before moving
        if (allocator && buffer && allocation)
            vmaDestroyBuffer(allocator, buffer, allocation);

        // Pilfer resources from the other object
        allocator = other.allocator;
        buffer = other.buffer;
        allocation = other.allocation;
        memory = other.memory;
        descBufferInfo = other.descBufferInfo;
        deviceAddress = other.deviceAddress;
        
        // Nullify the other object
        other.allocator = VK_NULL_HANDLE;
        other.buffer = VK_NULL_HANDLE;
        other.allocation = VK_NULL_HANDLE;
    }
    return *this;
}

// --- Main Constructor Refactored with VMA ---
Buffer::Buffer(const Context& context, const Type type, vk::DeviceSize size, const void* data, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryProps)
{
    allocator = context.getAllocator();

    // Auto-configure usage and memory flags for known types
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

    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(size);
    bufferInfo.setUsage(usage);

    VmaAllocationCreateInfo allocInfo = {};
    if (memoryProps & vk::MemoryPropertyFlagBits::eDeviceLocal)
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    
    else if (memoryProps & vk::MemoryPropertyFlagBits::eHostVisible)
        allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    
    if (data && (memoryProps & vk::MemoryPropertyFlagBits::eHostVisible))
        allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo allocationResultInfo;
    if (vmaCreateBuffer(allocator,
                        reinterpret_cast<const VkBufferCreateInfo*>(&bufferInfo),
                        &allocInfo,
                        reinterpret_cast<VkBuffer*>(&buffer),
                        &allocation,
                        &allocationResultInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer using VMA!");
    }

    this->memory = allocationResultInfo.deviceMemory;

    if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
        vk::BufferDeviceAddressInfo addressInfo{buffer};
        deviceAddress = context.getDevice().getBufferAddressKHR(&addressInfo);
    }

    descBufferInfo.setBuffer(buffer);
    descBufferInfo.setOffset(0);
    descBufferInfo.setRange(size);

    if (data && allocationResultInfo.pMappedData) {
        std::memcpy(allocationResultInfo.pMappedData, data, size);
        if (!(memoryProps & vk::MemoryPropertyFlagBits::eHostCoherent))
            vmaFlushAllocation(allocator, allocation, 0, size);
    }
}