#include "GPU/ImageInterop.h"

#include <array>
#include <stdexcept>

#include <cuda_runtime.h>

#include "GPU/Checks.h"
#include "Vulkan/Context.h"

namespace
{
struct AovFormat
{
    vk::Format vulkanFormat;
    cudaChannelFormatDesc cudaFormat;
};

const std::array<AovFormat, AovCount> AovFormats{{
    {vk::Format::eR32G32B32A32Sfloat, {32, 32, 32, 32, cudaChannelFormatKindFloat}},
    {vk::Format::eR8G8B8A8Unorm, {8, 8, 8, 8, cudaChannelFormatKindUnsigned}},
    {vk::Format::eR16G16B16A16Sfloat, {16, 16, 16, 16, cudaChannelFormatKindFloat}},
    {vk::Format::eR32Uint, {32, 0, 0, 0, cudaChannelFormatKindUnsigned}},
    {vk::Format::eR16G16B16A16Sfloat, {16, 16, 16, 16, cudaChannelFormatKindFloat}},
    {vk::Format::eR16G16B16A16Sfloat, {16, 16, 16, 16, cudaChannelFormatKindFloat}},
}};

vk::UniqueSemaphore createExportableTimelineSemaphore(const Context& context)
{
    vk::ExportSemaphoreCreateInfo exportInfo{};
    exportInfo.handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
    vk::SemaphoreTypeCreateInfo timelineInfo{};
    timelineInfo.pNext = &exportInfo;
    timelineInfo.semaphoreType = vk::SemaphoreType::eTimeline;
    timelineInfo.initialValue = 0;
    vk::SemaphoreCreateInfo createInfo{};
    createInfo.pNext = &timelineInfo;
    return context.getDevice().createSemaphoreUnique(createInfo);
}

cudaExternalSemaphore_t importSemaphore(const Context& context, const vk::Semaphore semaphore)
{
    vk::SemaphoreGetFdInfoKHR fdInfo{};
    fdInfo.semaphore = semaphore;
    fdInfo.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
    const int fd = context.getDevice().getSemaphoreFdKHR(fdInfo);

    cudaExternalSemaphoreHandleDesc handleDesc{};
    handleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    handleDesc.handle.fd = fd;
    cudaExternalSemaphore_t imported{};
    NR_GPU_CHECK(cudaImportExternalSemaphore(&imported, &handleDesc));
    return imported;
}
}

ImageInterop::ImageInterop(Context& context) : context(context)
{
    createSemaphores();
}

ImageInterop::~ImageInterop()
{
    destroyCudaImages();
    if (cudaRenderReady != nullptr)
        cudaDestroyExternalSemaphore(cudaRenderReady);
    if (cudaBufferReleased != nullptr)
        cudaDestroyExternalSemaphore(cudaBufferReleased);
}

void ImageInterop::createSemaphores()
{
    renderReady = createExportableTimelineSemaphore(context);
    bufferReleased = createExportableTimelineSemaphore(context);
    cudaRenderReady = importSemaphore(context, renderReady.get());
    cudaBufferReleased = importSemaphore(context, bufferReleased.get());
}

void ImageInterop::destroyCudaImages() noexcept
{
    for (auto& buffer : importedImages)
    {
        for (ImportedImage& imported : buffer)
        {
            if (imported.surface != 0)
                cudaDestroySurfaceObject(imported.surface);
            if (imported.memory != nullptr)
                cudaDestroyExternalMemory(imported.memory);
            imported = {};
        }
    }
}

void ImageInterop::resize(const uint32_t width, const uint32_t height)
{
    destroyCudaImages();
    for (auto& buffer : images)
        buffer.clear();

    constexpr vk::ImageUsageFlags usage =
        vk::ImageUsageFlagBits::eStorage |
        vk::ImageUsageFlagBits::eSampled |
        vk::ImageUsageFlagBits::eTransferSrc |
        vk::ImageUsageFlagBits::eTransferDst;

    for (uint32_t bufferIndex = 0; bufferIndex < 2; ++bufferIndex)
    {
        images[bufferIndex].reserve(AovCount);
        for (uint32_t aovIndex = 0; aovIndex < AovCount; ++aovIndex)
        {
            images[bufferIndex].emplace_back(
                context,
                width,
                height,
                AovFormats[aovIndex].vulkanFormat,
                usage,
                vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
            Image& image = images[bufferIndex].back();

            vk::MemoryGetFdInfoKHR fdInfo{};
            fdInfo.memory = image.getMemory();
            fdInfo.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
            const int fd = context.getDevice().getMemoryFdKHR(fdInfo);

            cudaExternalMemoryHandleDesc memoryDesc{};
            memoryDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
            memoryDesc.handle.fd = fd;
            memoryDesc.size = image.getAllocationSize();

            ImportedImage& imported = importedImages[bufferIndex][aovIndex];
            NR_GPU_CHECK(cudaImportExternalMemory(&imported.memory, &memoryDesc));

            cudaExternalMemoryMipmappedArrayDesc arrayDesc{};
            arrayDesc.offset = 0;
            arrayDesc.formatDesc = AovFormats[aovIndex].cudaFormat;
            arrayDesc.extent = make_cudaExtent(width, height, 0);
            arrayDesc.flags = cudaArraySurfaceLoadStore;
            arrayDesc.numLevels = 1;
            NR_GPU_CHECK(cudaExternalMemoryGetMappedMipmappedArray(
                &imported.mipmappedArray, imported.memory, &arrayDesc));
            NR_GPU_CHECK(cudaGetMipmappedArrayLevel(&imported.array, imported.mipmappedArray, 0));

            cudaResourceDesc resourceDesc{};
            resourceDesc.resType = cudaResourceTypeArray;
            resourceDesc.res.array.array = imported.array;
            NR_GPU_CHECK(cudaCreateSurfaceObject(&imported.surface, &resourceDesc));
        }
    }

    context.oneTimeSubmit([&](const vk::CommandBuffer commandBuffer)
    {
        for (auto& buffer : images)
            for (Image& image : buffer)
                image.setImageLayout(commandBuffer, vk::ImageLayout::eGeneral);
    });
}

OutputSurfaces ImageInterop::getSurfaces(const uint32_t bufferIndex) const
{
    if (bufferIndex >= 2 || images[bufferIndex].size() != AovCount)
        throw std::runtime_error("AOV interop images are not initialized");
    const auto& imported = importedImages[bufferIndex];
    return {
        imported[static_cast<uint32_t>(Aov::Color)].surface,
        imported[static_cast<uint32_t>(Aov::Albedo)].surface,
        imported[static_cast<uint32_t>(Aov::Normal)].surface,
        imported[static_cast<uint32_t>(Aov::Cryptomatte)].surface,
        imported[static_cast<uint32_t>(Aov::Position)].surface,
        imported[static_cast<uint32_t>(Aov::Material)].surface,
        images[bufferIndex][0].getWidth(),
        images[bufferIndex][0].getHeight()};
}

Image& ImageInterop::getImage(const uint32_t bufferIndex, const Aov aov)
{
    return images.at(bufferIndex).at(static_cast<uint32_t>(aov));
}

const Image& ImageInterop::getImage(const uint32_t bufferIndex, const Aov aov) const
{
    return images.at(bufferIndex).at(static_cast<uint32_t>(aov));
}

void ImageInterop::waitForRelease(const cudaStream_t stream, const uint64_t value) const
{
    cudaExternalSemaphoreWaitParams params{};
    params.params.fence.value = value;
    NR_GPU_CHECK(cudaWaitExternalSemaphoresAsync(&cudaBufferReleased, &params, 1, stream));
}

void ImageInterop::signalReady(const cudaStream_t stream, const uint64_t value) const
{
    cudaExternalSemaphoreSignalParams params{};
    params.params.fence.value = value;
    NR_GPU_CHECK(cudaSignalExternalSemaphoresAsync(&cudaRenderReady, &params, 1, stream));
}
