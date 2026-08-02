#pragma once

#include <cstddef>
#include <utility>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Backend/CUDA/rstd/Memory.h"

namespace nr::cuda
{

class UniqueAsyncDeviceBuffer
{
public:
    UniqueAsyncDeviceBuffer() = default;
    UniqueAsyncDeviceBuffer(std::size_t bytes, cudaStream_t ownerStream) { allocate(bytes, ownerStream); }
    ~UniqueAsyncDeviceBuffer() noexcept { reset(); }

    UniqueAsyncDeviceBuffer(const UniqueAsyncDeviceBuffer&) = delete;
    UniqueAsyncDeviceBuffer& operator=(const UniqueAsyncDeviceBuffer&) = delete;

    UniqueAsyncDeviceBuffer(UniqueAsyncDeviceBuffer&& other) noexcept
        : ptr(std::exchange(other.ptr, nullptr)),
          stream(std::exchange(other.stream, nullptr)),
          bytes(std::exchange(other.bytes, 0))
    {
    }

    UniqueAsyncDeviceBuffer& operator=(UniqueAsyncDeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            ptr = std::exchange(other.ptr, nullptr);
            stream = std::exchange(other.stream, nullptr);
            bytes = std::exchange(other.bytes, 0);
        }
        return *this;
    }

    void allocate(std::size_t newBytes, cudaStream_t ownerStream)
    {
        reset();
        stream = ownerStream;
        bytes = newBytes;
        ptr = nr::rstd::allocate_device(bytes, stream);
    }

    void reset() noexcept
    {
        if (ptr != nullptr)
            nr::rstd::deallocate_device(ptr, stream);
        ptr = nullptr;
        stream = nullptr;
        bytes = 0;
    }

    void* get() const noexcept { return ptr; }
    template <typename T>
    T* as() const noexcept { return static_cast<T*>(ptr); }
    CUdeviceptr devicePtr() const noexcept { return reinterpret_cast<CUdeviceptr>(ptr); }
    std::size_t size() const noexcept { return bytes; }
    explicit operator bool() const noexcept { return ptr != nullptr; }

private:
    void* ptr{};
    cudaStream_t stream{};
    std::size_t bytes{};
};

}
