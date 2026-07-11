// Originally derived from pbrt v4 pstd (Apache 2.0)
#pragma once

#include "CUDA/Annotations.h"
#include "CUDA/rstd/Allocator.h"

#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>

#if !defined(NR_BACKEND_CUDA)
#include <memory>
#include <vector>
#endif

namespace nr::rstd {

#if defined(NR_BACKEND_CUDA)

template <typename T, class Alloc = allocator<T>>
class vector {
public:
    vector() noexcept = default;

    explicit vector(const Alloc& alloc) noexcept : alloc_(alloc) {}

    explicit vector(std::size_t n, const Alloc& alloc = Alloc{}) : alloc_(alloc)
    {
        resize(n);
    }

    vector(std::size_t n, const T& value, const Alloc& alloc = Alloc{}) : alloc_(alloc)
    {
        reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            alloc_.construct(ptr_ + i, value);
        size_ = n;
    }

    template <typename It, typename = std::enable_if_t<!std::is_integral_v<It>>>
    vector(It first, It last, const Alloc& alloc = Alloc{}) : alloc_(alloc)
    {
        for (; first != last; ++first)
            push_back(*first);
    }

    vector(std::initializer_list<T> init, const Alloc& alloc = Alloc{}) : alloc_(alloc)
    {
        reserve(init.size());
        for (const T& v : init) {
            alloc_.construct(ptr_ + size_, v);
            ++size_;
        }
    }

    vector(const vector& other) : alloc_(other.alloc_)
    {
        reserve(other.size_);
        for (std::size_t i = 0; i < other.size_; ++i)
            alloc_.construct(ptr_ + i, other.ptr_[i]);
        size_ = other.size_;
    }

    vector(vector&& other) noexcept
        : alloc_(std::move(other.alloc_)), ptr_(other.ptr_), size_(other.size_), capacity_(other.capacity_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    vector& operator=(const vector& other)
    {
        if (this != &other) {
            clear();
            reserve(other.size_);
            for (std::size_t i = 0; i < other.size_; ++i)
                alloc_.construct(ptr_ + i, other.ptr_[i]);
            size_ = other.size_;
        }
        return *this;
    }

    vector& operator=(vector&& other) noexcept
    {
        if (this != &other) {
            destroy_storage();
            alloc_ = std::move(other.alloc_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.ptr_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    vector& operator=(std::initializer_list<T> init)
    {
        clear();
        reserve(init.size());
        for (const T& v : init) {
            alloc_.construct(ptr_ + size_, v);
            ++size_;
        }
        return *this;
    }

    ~vector() noexcept { destroy_storage(); }

    NR_CPU_GPU T& operator[](std::size_t i) noexcept { return ptr_[i]; }
    NR_CPU_GPU const T& operator[](std::size_t i) const noexcept { return ptr_[i]; }
    NR_CPU_GPU T& front() noexcept { return ptr_[0]; }
    NR_CPU_GPU const T& front() const noexcept { return ptr_[0]; }
    NR_CPU_GPU T& back() noexcept { return ptr_[size_ - 1]; }
    NR_CPU_GPU const T& back() const noexcept { return ptr_[size_ - 1]; }
    NR_CPU_GPU T* data() noexcept { return ptr_; }
    NR_CPU_GPU const T* data() const noexcept { return ptr_; }
    NR_CPU_GPU T* begin() noexcept { return ptr_; }
    NR_CPU_GPU const T* begin() const noexcept { return ptr_; }
    NR_CPU_GPU const T* cbegin() const noexcept { return ptr_; }
    NR_CPU_GPU T* end() noexcept { return ptr_ + size_; }
    NR_CPU_GPU const T* end() const noexcept { return ptr_ + size_; }
    NR_CPU_GPU const T* cend() const noexcept { return ptr_ + size_; }
    NR_CPU_GPU bool empty() const noexcept { return size_ == 0; }
    NR_CPU_GPU std::size_t size() const noexcept { return size_; }
    NR_CPU_GPU std::size_t capacity() const noexcept { return capacity_; }

    void clear() noexcept
    {
        for (std::size_t i = 0; i < size_; ++i)
            alloc_.destroy(ptr_ + i);
        size_ = 0;
    }

    void reserve(std::size_t new_capacity)
    {
        if (new_capacity <= capacity_)
            return;
        T* new_ptr = alloc_.allocate(new_capacity);
        for (std::size_t i = 0; i < size_; ++i) {
            alloc_.construct(new_ptr + i, std::move(ptr_[i]));
            alloc_.destroy(ptr_ + i);
        }
        if (ptr_)
            alloc_.deallocate(ptr_, capacity_);
        ptr_ = new_ptr;
        capacity_ = new_capacity;
    }

    void resize(std::size_t new_size)
    {
        if (new_size < size_) {
            for (std::size_t i = new_size; i < size_; ++i)
                alloc_.destroy(ptr_ + i);
            size_ = new_size;
            return;
        }
        if (new_size > size_) {
            reserve(new_size);
            for (std::size_t i = size_; i < new_size; ++i)
                alloc_.construct(ptr_ + i);
            size_ = new_size;
        }
    }

    void resize(std::size_t new_size, const T& value)
    {
        if (new_size < size_) {
            for (std::size_t i = new_size; i < size_; ++i)
                alloc_.destroy(ptr_ + i);
            size_ = new_size;
            return;
        }
        if (new_size > size_) {
            reserve(new_size);
            for (std::size_t i = size_; i < new_size; ++i)
                alloc_.construct(ptr_ + i, value);
            size_ = new_size;
        }
    }

    template <typename... Args>
    T& emplace_back(Args&&... args)
    {
        if (size_ == capacity_)
            reserve(capacity_ == 0 ? 4 : capacity_ * 2);
        alloc_.construct(ptr_ + size_, std::forward<Args>(args)...);
        ++size_;
        return ptr_[size_ - 1];
    }

    void push_back(const T& value) { emplace_back(value); }
    void push_back(T&& value) { emplace_back(std::move(value)); }

    void pop_back()
    {
        alloc_.destroy(ptr_ + size_ - 1);
        --size_;
    }

    NR_CPU_GPU bool operator==(const vector& rhs) const noexcept
    {
        if (size_ != rhs.size_)
            return false;
        for (std::size_t i = 0; i < size_; ++i)
            if (!(ptr_[i] == rhs.ptr_[i]))
                return false;
        return true;
    }

    NR_CPU_GPU bool operator!=(const vector& rhs) const noexcept { return !(*this == rhs); }

private:
    void destroy_storage() noexcept
    {
        clear();
        if (ptr_) {
            alloc_.deallocate(ptr_, capacity_);
            ptr_ = nullptr;
        }
        capacity_ = 0;
    }

    Alloc alloc_{};
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
};

#else

template <typename T, class Alloc = std::allocator<T>>
using vector = std::vector<T, Alloc>;

#endif

} // namespace nr::rstd
