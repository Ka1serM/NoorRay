#include "Buffer.h"
#include <stdexcept>

Buffer::Buffer(): allocator(nullptr), allocation(nullptr), deviceAddress(0)
{
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
      deviceAddress(other.deviceAddress),
      mappedData(other.mappedData)
{
    // Nullify the other object so its destructor does nothing
    other.allocator = VK_NULL_HANDLE;
    other.buffer = VK_NULL_HANDLE;
    other.allocation = VK_NULL_HANDLE;
    other.mappedData = nullptr;
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
        mappedData = other.mappedData;
        
        // Nullify the other object
        other.allocator = VK_NULL_HANDLE;
        other.buffer = VK_NULL_HANDLE;
        other.allocation = VK_NULL_HANDLE;
        other.mappedData = nullptr;
    }
    return *this;
}

Buffer::Buffer(const Context& context, const Type type, vk::DeviceSize size, const void* data, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryProps)
{
    allocator = context.getAllocator();

    // Auto-configure usage and memory flags for known types
    if (type != Type::Custom) {
        switch (type) {
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
    if (memoryProps & vk::MemoryPropertyFlagBits::eDeviceLocal) {
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    }
    else if (memoryProps & vk::MemoryPropertyFlagBits::eHostVisible) {
        // We need to distinguish between uploading to the GPU and reading back from it.
        if (usage & vk::BufferUsageFlagBits::eTransferDst) {
            // This buffer is the destination of a GPU transfer, so it's for readback.
            allocInfo.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
        } else {
            // Otherwise, assume it's for uploading data.
            allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        }
    }

    // Persistently map all host-visible buffers (zero-cost on modern drivers).
    if (memoryProps & vk::MemoryPropertyFlagBits::eHostVisible)
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
    this->mappedData = allocationResultInfo.pMappedData;

    if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
        vk::BufferDeviceAddressInfo addressInfo{buffer};
        deviceAddress = context.getDevice().getBufferAddressKHR(&addressInfo);
    }

    descBufferInfo.setBuffer(buffer);
    descBufferInfo.setOffset(0);
    descBufferInfo.setRange(size);

    if (data && allocationResultInfo.pMappedData) {
        std::memcpy(allocationResultInfo.pMappedData, data, size);
        // Flushing is only needed if memory is not HOST_COHERENT.
        // VMA_MEMORY_USAGE_CPU_TO_GPU and GPU_TO_CPU will typically select coherent memory anyway.
        if (!(memoryProps & vk::MemoryPropertyFlagBits::eHostCoherent))
            vmaFlushAllocation(allocator, allocation, 0, size);
    }
}
