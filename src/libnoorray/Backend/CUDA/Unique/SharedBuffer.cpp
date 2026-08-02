#include "Backend/CUDA/Unique/SharedBuffer.h"

#include <stdexcept>
#include <utility>
#include <unistd.h>

#include "Backend/CUDA/Checks.h"
#include "Backend/Vulkan/Runtime/Context.h"

namespace
{
constexpr VkExternalMemoryHandleTypeFlagBits ExternalHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

constexpr VkExportMemoryAllocateInfo ExportMemoryInfo{
    VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
    nullptr,
    ExternalHandleType};
}

nr::cuda::UniqueSharedBuffer::UniqueSharedBuffer(UniqueSharedBuffer&& other) noexcept
{
    *this = std::move(other);
}

nr::cuda::UniqueSharedBuffer& nr::cuda::UniqueSharedBuffer::operator=(
    UniqueSharedBuffer&& other) noexcept
{
    if (this == &other)
        return *this;

    reset();
    allocator_ = std::exchange(other.allocator_, nullptr);
    pool_ = std::exchange(other.pool_, nullptr);
    buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, nullptr);
    hostPointer_ = std::exchange(other.hostPointer_, nullptr);
    cudaMemory_ = std::exchange(other.cudaMemory_, nullptr);
    cudaPointer_ = std::exchange(other.cudaPointer_, nullptr);
    descriptorInfo_ = other.descriptorInfo_;
    other.descriptorInfo_ = vk::DescriptorBufferInfo{};
    return *this;
}

void nr::cuda::UniqueSharedBuffer::create(
    Context& context,
    const vk::DeviceSize bytes,
    const vk::BufferUsageFlags usage)
{
    if (bytes == 0)
        throw std::invalid_argument("A shared buffer cannot have zero size");

    reset();
    allocator_ = context.getAllocator();

    VkExternalMemoryBufferCreateInfo externalBufferInfo{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        nullptr,
        ExternalHandleType};
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.pNext = &externalBufferInfo;
    bufferInfo.size = bytes;
    bufferInfo.usage = static_cast<VkBufferUsageFlags>(usage);
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
        | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
        | VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationCreateInfo.requiredFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    uint32_t memoryTypeIndex{};
    if (vmaFindMemoryTypeIndexForBufferInfo(
            allocator_, &bufferInfo, &allocationCreateInfo, &memoryTypeIndex) != VK_SUCCESS)
    {
        reset();
        throw std::runtime_error("No host-coherent Vulkan memory type supports CUDA export");
    }

    VmaPoolCreateInfo poolInfo{};
    poolInfo.memoryTypeIndex = memoryTypeIndex;
    poolInfo.pMemoryAllocateNext = const_cast<VkExportMemoryAllocateInfo*>(&ExportMemoryInfo);
    if (vmaCreatePool(allocator_, &poolInfo, &pool_) != VK_SUCCESS)
    {
        reset();
        throw std::runtime_error("Failed to create VMA external-memory pool");
    }

    allocationCreateInfo.pool = pool_;
    VmaAllocationInfo allocationInfo{};
    if (vmaCreateBuffer(
            allocator_, &bufferInfo, &allocationCreateInfo,
            &buffer_, &allocation_, &allocationInfo) != VK_SUCCESS)
    {
        reset();
        throw std::runtime_error("Failed to create VMA CUDA-shared buffer");
    }
    hostPointer_ = allocationInfo.pMappedData;

    vk::MemoryGetFdInfoKHR fdInfo{};
    fdInfo.memory = allocationInfo.deviceMemory;
    fdInfo.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
    const int fd = context.getDevice().getMemoryFdKHR(fdInfo);

    cudaExternalMemoryHandleDesc memoryDesc{};
    memoryDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memoryDesc.handle.fd = fd;
    memoryDesc.size = allocationInfo.size;
    memoryDesc.flags = cudaExternalMemoryDedicated;
    const cudaError_t importResult = cudaImportExternalMemory(&cudaMemory_, &memoryDesc);
    if (importResult != cudaSuccess)
    {
        close(fd); // CUDA only takes ownership after a successful opaque-fd import.
        reset();
        NR_GPU_CHECK(importResult);
    }

    cudaExternalMemoryBufferDesc mappingDesc{};
    mappingDesc.size = bytes;
    const cudaError_t mappingResult =
        cudaExternalMemoryGetMappedBuffer(&cudaPointer_, cudaMemory_, &mappingDesc);
    if (mappingResult != cudaSuccess)
    {
        reset();
        NR_GPU_CHECK(mappingResult);
    }

    descriptorInfo_ = vk::DescriptorBufferInfo(vk::Buffer(buffer_), 0, bytes);
}

void nr::cuda::UniqueSharedBuffer::reset() noexcept
{
    if (cudaMemory_)
        cudaDestroyExternalMemory(cudaMemory_);
    cudaPointer_ = nullptr;
    cudaMemory_ = nullptr;

    if (allocator_ && buffer_ && allocation_)
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
    buffer_ = VK_NULL_HANDLE;
    allocation_ = nullptr;
    hostPointer_ = nullptr;

    if (allocator_ && pool_)
        vmaDestroyPool(allocator_, pool_);
    pool_ = nullptr;
    allocator_ = nullptr;
    descriptorInfo_ = vk::DescriptorBufferInfo{};
}
