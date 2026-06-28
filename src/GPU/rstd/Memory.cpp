#include "GPU/rstd/Memory.h"

#if defined(NR_BACKEND_CUDA)

#include "GPU/Memory.h"
#include <cuda_runtime.h>

namespace nr::rstd {

void* allocate_managed(std::size_t bytes)
{
    return nr::gpu::mallocManaged(bytes);
}

void deallocate_managed(void* p) noexcept
{
    if (p)
        cudaFree(p);
}

}

#endif
