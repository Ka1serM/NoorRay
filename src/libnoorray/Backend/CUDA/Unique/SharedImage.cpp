#include "Backend/CUDA/Unique/SharedImage.h"

#include <array>
#include <cuda_runtime.h>

#include "Backend/CUDA/Checks.h"
#include "Backend/Vulkan/Runtime/Context.h"

namespace
{
cudaChannelFormatDesc toCudaFormat(const vk::Format format)
{
    switch (format)
    {
    case vk::Format::eR32G32B32A32Sfloat: return cudaCreateChannelDesc<float4>();
    case vk::Format::eR8G8B8A8Unorm:      return {8, 8, 8, 8, cudaChannelFormatKindUnsigned};
    case vk::Format::eR16G16B16A16Sfloat: return {16, 16, 16, 16, cudaChannelFormatKindFloat};
    case vk::Format::eR32Uint:            return {32, 0, 0, 0, cudaChannelFormatKindUnsigned};
    default:                              return cudaCreateChannelDesc<float4>();
    }
}
}

void nr::cuda::UniqueSharedImage::create(
    Context& context,
    const uint32_t width,
    const uint32_t height,
    const vk::Format format)
{
    reset();
    // A zero-sized request leaves an empty image rather than reaching CUDA,
    // which rejects a zero extent with an exception that would otherwise cross
    // a host boundary (a Hydra callback, for instance).
    if (width == 0 || height == 0)
        return;

    constexpr vk::ImageUsageFlags usage =
        vk::ImageUsageFlagBits::eStorage |
        vk::ImageUsageFlagBits::eSampled |
        vk::ImageUsageFlagBits::eTransferSrc |
        vk::ImageUsageFlagBits::eTransferDst;

    image = Image(context, width, height, format, usage,
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);

    vk::MemoryGetFdInfoKHR fdInfo{};
    fdInfo.memory = image.getMemory();
    fdInfo.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
    const int fd = context.getDevice().getMemoryFdKHR(fdInfo);

    cudaExternalMemoryHandleDesc memoryDesc{};
    memoryDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memoryDesc.handle.fd = fd;
    memoryDesc.size = image.getAllocationSize();
    NR_GPU_CHECK(cudaImportExternalMemory(&cudaMemory, &memoryDesc));

    cudaExternalMemoryMipmappedArrayDesc arrayDesc{};
    arrayDesc.offset = 0;
    arrayDesc.formatDesc = toCudaFormat(format);
    arrayDesc.extent = make_cudaExtent(width, height, 0);
    arrayDesc.flags = cudaArraySurfaceLoadStore;
    arrayDesc.numLevels = 1;
    NR_GPU_CHECK(cudaExternalMemoryGetMappedMipmappedArray(
        &cudaMipmappedArray, cudaMemory, &arrayDesc));
    NR_GPU_CHECK(cudaGetMipmappedArrayLevel(&cudaArray, cudaMipmappedArray, 0));
    image.setCudaArray(cudaArray);

    cudaResourceDesc resourceDesc{};
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = cudaArray;
    NR_GPU_CHECK(cudaCreateSurfaceObject(&surface, &resourceDesc));

    context.oneTimeSubmit([&](const vk::CommandBuffer commandBuffer)
    {
        image.setImageLayout(commandBuffer, vk::ImageLayout::eGeneral);
    });
}

void nr::cuda::UniqueSharedImage::reset() noexcept
{
    if (surface != 0)
        cudaDestroySurfaceObject(surface);
    // The mipmapped array mapped onto the imported memory has to go first and
    // explicitly: cudaDestroyExternalMemory() does not free it, and while it
    // lives it holds a reference to the Vulkan allocation, so freeing the image
    // below would not actually give the memory back.
    if (cudaMipmappedArray != nullptr)
        cudaFreeMipmappedArray(cudaMipmappedArray);
    if (cudaMemory != nullptr)
        cudaDestroyExternalMemory(cudaMemory);
    image = Image{};
    surface = 0;
    cudaMemory = nullptr;
    cudaMipmappedArray = nullptr;
    cudaArray = nullptr;
}
