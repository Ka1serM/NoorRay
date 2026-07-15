#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "CUDA/Unique/SharedBuffer.h"

class Context;

namespace nr::cuda
{

template <typename T>
class SharedVector
{
    static_assert(std::is_trivially_copyable_v<T>);

public:
    SharedVector() = default;
    explicit SharedVector(
        Context& context,
        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer)
        : context_(&context), usage_(usage)
    {
    }

    T* data() { return static_cast<T*>(buffer_.hostPointer()); }
    const T* data() const { return static_cast<const T*>(buffer_.hostPointer()); }
    T* devicePointer() { return static_cast<T*>(buffer_.cudaPointer()); }
    const T* devicePointer() const { return static_cast<const T*>(buffer_.cudaPointer()); }

    T& operator[](std::size_t index) { return data()[index]; }
    const T& operator[](std::size_t index) const { return data()[index]; }
    T& back() { return data()[size_ - 1]; }
    const T& back() const { return data()[size_ - 1]; }

    std::size_t size() const { return size_; }
    std::size_t capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }
    const UniqueSharedBuffer& buffer() const { return buffer_; }

    void clear() { size_ = 0; }

    void reserve(const std::size_t newCapacity)
    {
        if (newCapacity <= capacity_)
            return;
        if (!context_)
            throw std::logic_error("SharedVector has no Context");

        UniqueSharedBuffer newBuffer;
        newBuffer.create(*context_, newCapacity * sizeof(T), usage_);
        if (size_ != 0)
            std::memcpy(newBuffer.hostPointer(), buffer_.hostPointer(), size_ * sizeof(T));
        buffer_ = std::move(newBuffer);
        capacity_ = newCapacity;
    }

    T& push_back(const T& value)
    {
        if (size_ == capacity_)
            reserve(capacity_ == 0 ? 4 : capacity_ * 2);
        T& result = data()[size_++];
        result = value;
        return result;
    }

    void pop_back() { --size_; }

private:
    Context* context_{};
    vk::BufferUsageFlags usage_{vk::BufferUsageFlagBits::eStorageBuffer};
    UniqueSharedBuffer buffer_;
    std::size_t size_{};
    std::size_t capacity_{};
};

}
