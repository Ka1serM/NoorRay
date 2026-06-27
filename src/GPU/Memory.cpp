#include "GPU/Memory.h"

#include "GPU/Checks.h"

namespace nr::gpu
{
void* mallocDevice(const std::size_t bytes, cudaStream_t stream)
{
    if (bytes == 0)
        return nullptr;

    void* pointer = nullptr;
    NR_GPU_CHECK(cudaMallocAsync(&pointer, bytes, stream));
    return pointer;
}

void freeDevice(void* pointer, cudaStream_t stream) noexcept
{
    if (pointer != nullptr)
        cudaFreeAsync(pointer, stream);
}

void* mallocManaged(const std::size_t bytes)
{
    if (bytes == 0)
        return nullptr;

    void* pointer = nullptr;
    NR_GPU_CHECK(cudaMallocManaged(&pointer, bytes));
    return pointer;
}
}
