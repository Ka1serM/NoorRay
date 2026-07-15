#pragma once

#include <cuda_runtime_api.h>

#include "CUDA/Unique/Check.h"
#include "CUDA/Unique/Handle.h"

namespace nr::cuda
{

class UniqueStream
{
public:
    UniqueStream() = default;
    ~UniqueStream() noexcept = default;

    UniqueStream(const UniqueStream&) = delete;
    UniqueStream& operator=(const UniqueStream&) = delete;

    UniqueStream(UniqueStream&&) noexcept = default;
    UniqueStream& operator=(UniqueStream&&) noexcept = default;

    void createWithPriority(unsigned int flags, int priority)
    {
        reset();
        NR_CUDA_UNIQUE_CHECK(cudaStreamCreateWithPriority(stream.put(), flags, priority));
    }

    void reset() noexcept
    {
        stream.reset();
    }

    cudaStream_t get() const noexcept { return stream.get(); }
    explicit operator bool() const noexcept { return static_cast<bool>(stream); }

private:
    UniqueHandle<cudaStream_t, cudaStreamDestroy> stream;
};

}
