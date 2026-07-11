#pragma once

#include <utility>

#include <cuda_runtime_api.h>

#include "CUDA/Unique/Check.h"

namespace nr::cuda
{

class UniqueStream
{
public:
    UniqueStream() = default;
    ~UniqueStream() noexcept { reset(); }

    UniqueStream(const UniqueStream&) = delete;
    UniqueStream& operator=(const UniqueStream&) = delete;

    UniqueStream(UniqueStream&& other) noexcept
        : stream(std::exchange(other.stream, nullptr))
    {
    }

    UniqueStream& operator=(UniqueStream&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            stream = std::exchange(other.stream, nullptr);
        }
        return *this;
    }

    void createWithPriority(unsigned int flags, int priority)
    {
        reset();
        NR_CUDA_UNIQUE_CHECK(cudaStreamCreateWithPriority(&stream, flags, priority));
    }

    void reset() noexcept
    {
        if (stream != nullptr)
            cudaStreamDestroy(stream);
        stream = nullptr;
    }

    cudaStream_t get() const noexcept { return stream; }
    explicit operator bool() const noexcept { return stream != nullptr; }

private:
    cudaStream_t stream{};
};

}
