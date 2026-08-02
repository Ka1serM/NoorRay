#include "Backend/CUDA/rstd/Memory.h"
#include "Backend/CUDA/UnifiedMemoryObject.h"

#if defined(NR_BACKEND_CUDA)

#include "Backend/CUDA/Checks.h"
#include <cuda_runtime.h>

namespace nr::rstd {

void* allocate_device(const std::size_t bytes, cudaStream_t stream)
{
    if (bytes == 0)
        return nullptr;

    void* p = nullptr;
    NR_GPU_CHECK(cudaMallocAsync(&p, bytes, stream));
    return p;
}

void deallocate_device(void* p, cudaStream_t stream) noexcept
{
    if (p)
        cudaFreeAsync(p, stream);
}

void* allocate_managed(const std::size_t bytes)
{
    if (bytes == 0)
        return nullptr;

    void* p = nullptr;
    NR_GPU_CHECK(cudaMallocManaged(&p, bytes));
    return p;
}

void deallocate_managed(void* p) noexcept
{
    if (p)
        cudaFree(p);
}

}

void* nr::UnifiedMemoryObject::operator new(const std::size_t size)
{
    return rstd::allocate_managed(size);
}

void nr::UnifiedMemoryObject::operator delete(void* pointer) noexcept
{
    rstd::deallocate_managed(pointer);
}

void nr::UnifiedMemoryObject::operator delete(void* pointer, std::size_t) noexcept
{
    rstd::deallocate_managed(pointer);
}

#endif
