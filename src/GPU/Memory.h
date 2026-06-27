#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace nr::gpu
{
void* mallocDevice(std::size_t bytes, cudaStream_t stream);
void freeDevice(void* pointer, cudaStream_t stream) noexcept;
void* mallocManaged(std::size_t bytes);
}
