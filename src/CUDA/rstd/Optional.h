#pragma once

#if defined(NR_BACKEND_CUDA)
#include <cuda/std/optional>
#else
#include <optional>
#endif

namespace nr::rstd {

#if defined(NR_BACKEND_CUDA)
template <typename T> using optional = cuda::std::optional<T>;
using nullopt_t = cuda::std::nullopt_t;
inline constexpr nullopt_t nullopt = cuda::std::nullopt;
#else
template <typename T> using optional = std::optional<T>;
using nullopt_t = std::nullopt_t;
inline constexpr nullopt_t nullopt = std::nullopt;
#endif

}
