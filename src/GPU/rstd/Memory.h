#pragma once

#include <cstddef>

#if defined(NR_BACKEND_CUDA)

namespace nr::rstd {

void* allocate_managed(std::size_t bytes);
void  deallocate_managed(void* p) noexcept;

}

#endif
