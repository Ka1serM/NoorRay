#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

#include "Kernels/Output.h"
#include "Vulkan/Image.h"

class Context;

enum class Aov : uint32_t
{
    Color,
    Albedo,
    Normal,
    Cryptomatte,
    Position,
    Material,
    Count
};

static constexpr uint32_t AovCount = static_cast<uint32_t>(Aov::Count);

class ImageInterop
{
public:
    explicit ImageInterop(Context& context);
    ~ImageInterop();

    ImageInterop(const ImageInterop&) = delete;
    ImageInterop& operator=(const ImageInterop&) = delete;

    void resize(uint32_t width, uint32_t height);
    OutputSurfaces getSurfaces(uint32_t bufferIndex) const;
    Image& getImage(uint32_t bufferIndex, Aov aov);
    const Image& getImage(uint32_t bufferIndex, Aov aov) const;

    void waitForRelease(cudaStream_t stream, uint64_t value) const;
    void signalReady(cudaStream_t stream, uint64_t value) const;

    vk::Semaphore getRenderReadySemaphore() const { return renderReady.get(); }
    vk::Semaphore getBufferReleasedSemaphore() const { return bufferReleased.get(); }

private:
    struct ImportedImage
    {
        cudaExternalMemory_t memory{};
        cudaMipmappedArray_t mipmappedArray{};
        cudaArray_t array{};
        cudaSurfaceObject_t surface{};
    };

    Context& context;
    std::array<std::vector<Image>, 2> images;
    std::array<std::array<ImportedImage, AovCount>, 2> importedImages{};
    vk::UniqueSemaphore renderReady;
    vk::UniqueSemaphore bufferReleased;
    cudaExternalSemaphore_t cudaRenderReady{};
    cudaExternalSemaphore_t cudaBufferReleased{};

    void createSemaphores();
    void destroyCudaImages() noexcept;
};
