// Originally derived from pbrt v4 pstd (Apache 2.0)
#pragma once

#include "CUDA/Annotations.h"
#include "CUDA/UnifiedMemoryObject.h"
#include "CUDA/rstd/Allocator.h"

#include <cstddef>
#include <type_traits>
#include <utility>

#if !defined(NR_BACKEND_CUDA)
#include <memory>
#endif

namespace nr::rstd {

#if defined(NR_BACKEND_CUDA)

template <typename T, class Alloc = allocator<T>>
class unique_ptr {
public:
    NR_CPU_GPU constexpr unique_ptr() noexcept = default;
    NR_CPU_GPU constexpr unique_ptr(std::nullptr_t) noexcept {}
    NR_CPU_GPU explicit unique_ptr(T* p, Alloc alloc = Alloc{}) noexcept : ptr_(p), alloc_(alloc) {}

    unique_ptr(const unique_ptr&) = delete;
    unique_ptr& operator=(const unique_ptr&) = delete;

    NR_CPU_GPU unique_ptr(unique_ptr&& other) noexcept : ptr_(other.ptr_), alloc_(other.alloc_)
    {
        other.ptr_ = nullptr;
    }

    NR_CPU_GPU unique_ptr& operator=(unique_ptr&& other) noexcept
    {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            alloc_ = other.alloc_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    NR_CPU_GPU ~unique_ptr() noexcept { reset(); }

    NR_CPU_GPU T& operator*() const noexcept { return *ptr_; }
    NR_CPU_GPU T* operator->() const noexcept { return ptr_; }
    NR_CPU_GPU T* get() const noexcept { return ptr_; }
    NR_CPU_GPU explicit operator bool() const noexcept { return ptr_ != nullptr; }

    NR_CPU_GPU T* release() noexcept
    {
        T* p = ptr_;
        ptr_ = nullptr;
        return p;
    }

    NR_CPU_GPU void reset(T* p = nullptr) noexcept
    {
        if (ptr_) {
            if constexpr (std::is_base_of_v<UnifiedMemoryObject, T>) {
                delete ptr_;
            } else {
                alloc_.destroy(ptr_);
                alloc_.deallocate(ptr_, 1);
            }
        }
        ptr_ = p;
    }

private:
    T* ptr_ = nullptr;
    Alloc alloc_{};
};

// array specialization
template <typename T, class Alloc>
class unique_ptr<T[], Alloc> {
public:
    NR_CPU_GPU constexpr unique_ptr() noexcept = default;
    NR_CPU_GPU constexpr unique_ptr(std::nullptr_t) noexcept {}
    NR_CPU_GPU explicit unique_ptr(T* p, std::size_t n, Alloc alloc = Alloc{}) noexcept
        : ptr_(p), size_(n), alloc_(alloc) {}

    unique_ptr(const unique_ptr&) = delete;
    unique_ptr& operator=(const unique_ptr&) = delete;

    NR_CPU_GPU unique_ptr(unique_ptr&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), alloc_(other.alloc_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    NR_CPU_GPU unique_ptr& operator=(unique_ptr&& other) noexcept
    {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            size_ = other.size_;
            alloc_ = other.alloc_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    NR_CPU_GPU ~unique_ptr() noexcept { reset(); }

    NR_CPU_GPU T& operator[](std::size_t i) const noexcept { return ptr_[i]; }
    NR_CPU_GPU T* get() const noexcept { return ptr_; }
    NR_CPU_GPU std::size_t size() const noexcept { return size_; }
    NR_CPU_GPU explicit operator bool() const noexcept { return ptr_ != nullptr; }

    NR_CPU_GPU T* release() noexcept
    {
        T* p = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        return p;
    }

    NR_CPU_GPU void reset(T* p = nullptr, std::size_t n = 0) noexcept
    {
        if (ptr_) {
            for (std::size_t i = 0; i < size_; ++i)
                alloc_.destroy(ptr_ + i);
            alloc_.deallocate(ptr_, size_);
        }
        ptr_ = p;
        size_ = n;
    }

private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
    Alloc alloc_{};
};

template <typename T, typename... Args, std::enable_if_t<!std::is_array_v<T>, int> = 0>
NR_CPU_GPU inline unique_ptr<T> make_unique(Args&&... args)
{
    allocator<T> alloc;
    T* p = alloc.allocate(1);
    alloc.construct(p, std::forward<Args>(args)...);
    return unique_ptr<T>(p, alloc);
}

template <typename T, std::enable_if_t<std::is_unbounded_array_v<T>, int> = 0>
NR_CPU_GPU inline unique_ptr<T> make_unique(std::size_t n)
{
    using U = std::remove_extent_t<T>;
    allocator<U> alloc;
    U* p = alloc.allocate(n);
    for (std::size_t i = 0; i < n; ++i)
        alloc.construct(p + i);
    return unique_ptr<T>(p, n, alloc);
}

template <typename T, std::enable_if_t<std::is_bounded_array_v<T>, int> = 0>
void make_unique() = delete;

#else

template <typename T, class D = std::default_delete<T>>
using unique_ptr = std::unique_ptr<T, D>;

using std::make_unique;

#endif

} // namespace nr::rstd
