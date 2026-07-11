#pragma once

#include <utility>

#include <cuda_runtime_api.h>

#include "Vulkan/Buffer.h"

class Context;

namespace nr::cuda
{

// A host-visible Vulkan buffer imported into CUDA via an opaque-fd external memory
// handle. The same physical memory is reachable from three places at once: CPU
// (Buffer's persistently-mapped pointer), CUDA/OptiX (getDevicePointer), and Vulkan
// shaders (getBuffer() for descriptor binding). Writes from any side are visible to
// the others without an explicit copy, since the memory is host-coherent.
class UniqueSharedBuffer
{
public:
    UniqueSharedBuffer() = default;
    ~UniqueSharedBuffer() noexcept { reset(); }

    UniqueSharedBuffer(const UniqueSharedBuffer&) = delete;
    UniqueSharedBuffer& operator=(const UniqueSharedBuffer&) = delete;

    UniqueSharedBuffer(UniqueSharedBuffer&& other) noexcept
        : buffer(std::move(other.buffer)),
          cudaMemory(other.cudaMemory),
          cudaPointer(other.cudaPointer)
    {
        other.cudaMemory = nullptr;
        other.cudaPointer = nullptr;
    }

    UniqueSharedBuffer& operator=(UniqueSharedBuffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            buffer = std::move(other.buffer);
            cudaMemory = other.cudaMemory;
            cudaPointer = other.cudaPointer;
            other.cudaMemory = nullptr;
            other.cudaPointer = nullptr;
        }
        return *this;
    }

    void create(Context& context, vk::DeviceSize bytes, vk::BufferUsageFlags usage);
    void reset() noexcept;

    Buffer& getBuffer() { return buffer; }
    const Buffer& getBuffer() const { return buffer; }
    void* getHostPointer() const { return buffer.getMappedData(); }
    void* getDevicePointer() const { return cudaPointer; }
    vk::DeviceSize getSize() const { return buffer.getSize(); }

private:
    Buffer buffer;
    cudaExternalMemory_t cudaMemory{};
    void* cudaPointer{};
};

}
