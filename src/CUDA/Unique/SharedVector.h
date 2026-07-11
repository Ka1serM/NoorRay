#pragma once

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "CUDA/Unique/SharedBuffer.h"

class Context;

namespace nr::cuda
{

// A growable array of trivially-copyable elements backed by a single UniqueSharedBuffer.
// CPU code indexes it like a normal vector (data() / operator[]); GPU kernels read the
// same bytes through devicePointer(); Vulkan can bind getBuffer() as a descriptor.
// All three views stay in sync without any copy, since the backing memory is
// host-coherent and shared via an imported Vulkan allocation.
template <typename T>
class SharedVector
{
    static_assert(std::is_trivially_copyable_v<T>,
        "SharedVector only supports trivially-copyable GPU-resident types");

public:
    SharedVector() = default;
    explicit SharedVector(Context& context,
        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer)
        : context(&context), usage(usage)
    {
    }

    T* data() { return static_cast<T*>(buffer.getHostPointer()); }
    const T* data() const { return static_cast<const T*>(buffer.getHostPointer()); }
    T* devicePointer() { return static_cast<T*>(buffer.getDevicePointer()); }
    const T* devicePointer() const { return static_cast<const T*>(buffer.getDevicePointer()); }

    T& operator[](std::size_t i) { return data()[i]; }
    const T& operator[](std::size_t i) const { return data()[i]; }
    T& front() { return data()[0]; }
    T& back() { return data()[size_ - 1]; }
    const T& back() const { return data()[size_ - 1]; }

    std::size_t size() const { return size_; }
    std::size_t capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }

    Buffer& getBuffer() { return buffer.getBuffer(); }
    const Buffer& getBuffer() const { return buffer.getBuffer(); }

    void clear() { size_ = 0; }

    void reserve(const std::size_t newCapacity)
    {
        if (newCapacity <= capacity_)
            return;
        UniqueSharedBuffer newBuffer;
        newBuffer.create(*context, newCapacity * sizeof(T), usage);
        if (size_ > 0)
            std::memcpy(newBuffer.getHostPointer(), buffer.getHostPointer(), size_ * sizeof(T));
        buffer = std::move(newBuffer);
        capacity_ = newCapacity;
    }

    T& push_back(const T& value)
    {
        if (size_ == capacity_)
            reserve(capacity_ == 0 ? 4 : capacity_ * 2);
        T& slot = data()[size_];
        slot = value;
        ++size_;
        return slot;
    }

    void pop_back() { --size_; }

private:
    Context* context{};
    vk::BufferUsageFlags usage{vk::BufferUsageFlagBits::eStorageBuffer};
    UniqueSharedBuffer buffer;
    std::size_t size_{};
    std::size_t capacity_{};
};

}
