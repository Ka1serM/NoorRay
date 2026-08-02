#pragma once

#include <cstddef>
#include <utility>

#include <cuda.h>

#include "Backend/CUDA/rstd/Memory.h"

namespace nr::cuda
{

class UniqueManagedBuffer
{
public:
    UniqueManagedBuffer() = default;
    explicit UniqueManagedBuffer(std::size_t bytes) { allocate(bytes); }
    ~UniqueManagedBuffer() noexcept { reset(); }

    UniqueManagedBuffer(const UniqueManagedBuffer&) = delete;
    UniqueManagedBuffer& operator=(const UniqueManagedBuffer&) = delete;

    UniqueManagedBuffer(UniqueManagedBuffer&& other) noexcept
        : ptr(std::exchange(other.ptr, nullptr)),
          bytes(std::exchange(other.bytes, 0))
    {
    }

    UniqueManagedBuffer& operator=(UniqueManagedBuffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            ptr = std::exchange(other.ptr, nullptr);
            bytes = std::exchange(other.bytes, 0);
        }
        return *this;
    }

    void allocate(std::size_t newBytes)
    {
        reset();
        bytes = newBytes;
        ptr = nr::rstd::allocate_managed(bytes);
    }

    void reset() noexcept
    {
        if (ptr != nullptr)
            nr::rstd::deallocate_managed(ptr);
        ptr = nullptr;
        bytes = 0;
    }

    void* get() const noexcept { return ptr; }
    CUdeviceptr devicePtr() const noexcept { return reinterpret_cast<CUdeviceptr>(ptr); }
    std::size_t size() const noexcept { return bytes; }
    explicit operator bool() const noexcept { return ptr != nullptr; }

private:
    void* ptr{};
    std::size_t bytes{};
};

}
