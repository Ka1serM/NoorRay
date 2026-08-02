#pragma once

#include <cstddef>

#if defined(NR_BACKEND_CUDA)

#include <cuda_runtime_api.h>

namespace nr::rstd {

void* allocate_device(std::size_t bytes, cudaStream_t stream);
void  deallocate_device(void* p, cudaStream_t stream) noexcept;
void* allocate_managed(std::size_t bytes);
void  deallocate_managed(void* p) noexcept;

template <typename T>
T* allocate_device(const size_t count, const cudaStream_t stream)
{
    return static_cast<T*>(allocate_device(sizeof(T) * count, stream));
}

}

#endif
