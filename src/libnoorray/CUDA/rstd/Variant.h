#pragma once

#include "CUDA/Annotations.h"

#if defined(NR_BACKEND_CUDA)
#include <cuda/std/variant>
#else
#include <variant>
#endif

#include <utility>

namespace nr::rstd {

#if defined(NR_BACKEND_CUDA)
template <typename... Ts> using variant = ::cuda::std::variant<Ts...>;

template <typename Visitor, typename... Variants>
NR_CPU_GPU constexpr decltype(auto) visit(Visitor&& visitor, Variants&&... variants)
{
    return ::cuda::std::visit(std::forward<Visitor>(visitor), std::forward<Variants>(variants)...);
}
#else
template <typename... Ts> using variant = std::variant<Ts...>;

template <typename Visitor, typename... Variants>
constexpr decltype(auto) visit(Visitor&& visitor, Variants&&... variants)
{
    return std::visit(std::forward<Visitor>(visitor), std::forward<Variants>(variants)...);
}
#endif

}
