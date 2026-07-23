#pragma once

#if defined(NR_BACKEND_CUDA)
#include <cuda/std/span>
#else
#include <span>
#endif

namespace nr::rstd {

#if defined(NR_BACKEND_CUDA)
template <typename T> using span = ::cuda::std::span<T>;
#else
template <typename T> using span = std::span<T>;
#endif

}
