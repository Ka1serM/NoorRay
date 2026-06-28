#pragma once

#if defined(NR_BACKEND_CUDA)
#include <cuda/std/utility>
#else
#include <utility>
#endif

namespace nr::rstd {

#if defined(NR_BACKEND_CUDA)
template <typename T1, typename T2> using pair = cuda::std::pair<T1, T2>;
#else
template <typename T1, typename T2> using pair = std::pair<T1, T2>;
#endif

}
