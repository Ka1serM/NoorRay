#include "CUDA/Unique/SharedBuffer.h"

#include "CUDA/Checks.h"
#include "Vulkan/Context.h"

void nr::cuda::UniqueSharedBuffer::create(
    Context& context,
    const vk::DeviceSize bytes,
    const vk::BufferUsageFlags usage)
{
    reset();

    buffer = Buffer(context, bytes, usage, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);

    vk::MemoryGetFdInfoKHR fdInfo{};
    fdInfo.memory = buffer.getMemory();
    fdInfo.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
    const int fd = context.getDevice().getMemoryFdKHR(fdInfo);

    cudaExternalMemoryHandleDesc memoryDesc{};
    memoryDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memoryDesc.handle.fd = fd;
    memoryDesc.size = bytes;
    NR_GPU_CHECK(cudaImportExternalMemory(&cudaMemory, &memoryDesc));

    cudaExternalMemoryBufferDesc bufferDesc{};
    bufferDesc.offset = 0;
    bufferDesc.size = bytes;
    NR_GPU_CHECK(cudaExternalMemoryGetMappedBuffer(&cudaPointer, cudaMemory, &bufferDesc));
}

void nr::cuda::UniqueSharedBuffer::reset() noexcept
{
    // The mapped pointer returned by cudaExternalMemoryGetMappedBuffer does not need
    // (and must not be) freed separately; it is invalidated by cudaDestroyExternalMemory.
    if (cudaMemory != nullptr)
        cudaDestroyExternalMemory(cudaMemory);
    buffer = Buffer{};
    cudaPointer = nullptr;
    cudaMemory = nullptr;
}
