#pragma once

#include <cuda_runtime_api.h>

#include "Vulkan/Image.h"

class Context;

class SharedImage
{
public:
    SharedImage() = default;
    ~SharedImage() noexcept { destroy(); }

    SharedImage(const SharedImage&) = delete;
    SharedImage& operator=(const SharedImage&) = delete;

    SharedImage(SharedImage&& other) noexcept
        : image(std::move(other.image)),
          cudaMemory(other.cudaMemory),
          cudaMipmappedArray(other.cudaMipmappedArray),
          cudaArray(other.cudaArray),
          surface(other.surface)
    {
        other.cudaMemory = nullptr;
        other.cudaMipmappedArray = nullptr;
        other.cudaArray = nullptr;
        other.surface = 0;
    }

    SharedImage& operator=(SharedImage&& other) noexcept
    {
        if (this != &other)
        {
            destroy();
            image = std::move(other.image);
            cudaMemory = other.cudaMemory;
            cudaMipmappedArray = other.cudaMipmappedArray;
            cudaArray = other.cudaArray;
            surface = other.surface;
            other.cudaMemory = nullptr;
            other.cudaMipmappedArray = nullptr;
            other.cudaArray = nullptr;
            other.surface = 0;
        }
        return *this;
    }

    void create(Context& context, uint32_t width, uint32_t height, vk::Format format);
    void destroy() noexcept;

    Image& getImage() { return image; }
    const Image& getImage() const { return image; }
    cudaSurfaceObject_t getSurface() const { return surface; }

private:
    Image image;
    cudaExternalMemory_t cudaMemory{};
    cudaMipmappedArray_t cudaMipmappedArray{};
    cudaArray_t cudaArray{};
    cudaSurfaceObject_t surface{};
};
