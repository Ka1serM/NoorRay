// Originally derived from pbrt v4 pstd (Apache 2.0)
#pragma once

#include "GPU/rstd/Memory.h"

#include <cstddef>
#include <new>
#include <utility>

#if defined(NR_BACKEND_CUDA)

namespace nr::rstd {

template <typename T>
struct allocator {
    using value_type = T;

    allocator() noexcept = default;

    template <class U>
    allocator(const allocator<U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n)
    {
        return static_cast<T*>(allocate_managed(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t) noexcept
    {
        deallocate_managed(p);
    }

    template <class U, class... Args>
    void construct(U* p, Args&&... args)
    {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <class U>
    void destroy(U* p) noexcept
    {
        p->~U();
    }
};

}

#endif
