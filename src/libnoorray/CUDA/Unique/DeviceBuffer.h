#pragma once

#include <cstddef>
#include <utility>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "CUDA/Checks.h"

namespace nr::cuda
{

class UniqueDeviceBuffer
{
public:
    UniqueDeviceBuffer() = default;
    explicit UniqueDeviceBuffer(std::size_t bytes) { allocate(bytes); }
    ~UniqueDeviceBuffer() noexcept { reset(); }

    UniqueDeviceBuffer(const UniqueDeviceBuffer&) = delete;
    UniqueDeviceBuffer& operator=(const UniqueDeviceBuffer&) = delete;

    UniqueDeviceBuffer(UniqueDeviceBuffer&& other) noexcept
        : ptr(std::exchange(other.ptr, nullptr))
    {
    }

    UniqueDeviceBuffer& operator=(UniqueDeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            ptr = std::exchange(other.ptr, nullptr);
        }
        return *this;
    }

    void allocate(std::size_t bytes)
    {
        reset();
        NR_GPU_CHECK(cudaMalloc(&ptr, bytes));
    }

    void reset() noexcept
    {
        if (ptr != nullptr)
            cudaFree(ptr);
        ptr = nullptr;
    }

    void* get() const noexcept { return ptr; }
    CUdeviceptr devicePtr() const noexcept { return reinterpret_cast<CUdeviceptr>(ptr); }
    explicit operator bool() const noexcept { return ptr != nullptr; }

private:
    void* ptr{};
};

}
