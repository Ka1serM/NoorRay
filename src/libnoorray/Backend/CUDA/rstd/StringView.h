#pragma once

#if defined(NR_BACKEND_CUDA)
#include <cuda/std/string_view>
#else
#include <string_view>
#endif

namespace nr::rstd {

#if defined(NR_BACKEND_CUDA)
using string_view = ::cuda::std::string_view;
#else
using string_view = std::string_view;
#endif

}
