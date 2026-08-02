#pragma once

#include "Backend/CUDA/Annotations.h"

#if defined(NR_BACKEND_CUDA)
#include <cuda/std/tuple>
#else
#include <tuple>
#endif

#include <cstddef>
#include <utility>

namespace nr::rstd {

#if defined(NR_BACKEND_CUDA)
template <typename... Ts> using tuple = ::cuda::std::tuple<Ts...>;

template <std::size_t I, typename TupleLike>
NR_CPU_GPU constexpr decltype(auto) get(TupleLike&& t)
    noexcept(noexcept(::cuda::std::get<I>(std::forward<TupleLike>(t))))
{
    return ::cuda::std::get<I>(std::forward<TupleLike>(t));
}
#else
template <typename... Ts> using tuple = std::tuple<Ts...>;

template <std::size_t I, typename TupleLike>
constexpr decltype(auto) get(TupleLike&& t)
    noexcept(noexcept(std::get<I>(std::forward<TupleLike>(t))))
{
    return std::get<I>(std::forward<TupleLike>(t));
}
#endif

}
