#pragma once

#include "device.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <limits>
#include <type_traits>

namespace gpu {

namespace detail { struct BufferImpl; }

template<class T>
class Buffer {
public:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) noexcept = default;
    Buffer& operator=(Buffer&&) noexcept = default;

    GpuPtr<T> ptr() const;
    void upload(std::span<const T> data, std::size_t offset = 0);
    void download(std::span<T> destination) const;
    std::size_t size() const noexcept { return count_; }
    std::size_t byte_size() const noexcept { return count_ * sizeof(T); }
    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
    friend class Device;
    template<class U> friend class Buffer;
    Buffer(std::shared_ptr<detail::BufferImpl> impl, std::size_t count)
        : impl_(std::move(impl)), count_(count) {}

    std::shared_ptr<detail::BufferImpl> impl_;
    std::size_t count_ = 0;
};

} // namespace gpu

namespace gpu::detail {
std::shared_ptr<BufferImpl> make_buffer(const std::shared_ptr<DeviceImpl>&, std::size_t, std::size_t);
void upload_buffer(const std::shared_ptr<BufferImpl>&, const void*, std::size_t, std::size_t);
void download_buffer(const std::shared_ptr<BufferImpl>&, void*, std::size_t);
std::uint64_t buffer_address(const std::shared_ptr<BufferImpl>&);
}

namespace gpu {
template<class T>
GpuPtr<T> Buffer<T>::ptr() const {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "gpu::Buffer is empty");
    return {detail::buffer_address(impl_)};
}

template<class T>
Buffer<T> Device::buffer(const std::size_t count) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        throw Error(ErrorCode::InvalidArgument, "GPU buffer size overflows address space");
    if (count == 0)
        throw Error(ErrorCode::InvalidArgument, "gpu::Device::buffer requires a non-zero count");
    return Buffer<T>(detail::make_buffer(impl_, count * sizeof(T), alignof(T)), count);
}

template<class T>
void Buffer<T>::upload(std::span<const T> data, std::size_t offset) {
    if (!impl_) throw Error(ErrorCode::InvalidResource, "cannot upload to an empty buffer");
    if (offset > size() || data.size() > size() - offset)
        throw Error(ErrorCode::InvalidArgument, "upload range exceeds buffer");
    if (!data.empty())
        detail::upload_buffer(impl_, data.data(), data.size_bytes(), offset * sizeof(T));
}

template<class T>
void Buffer<T>::download(std::span<T> destination) const {
    if (!impl_) throw Error(ErrorCode::InvalidResource, "cannot download from an empty buffer");
    if (destination.size() > size())
        throw Error(ErrorCode::InvalidArgument, "download range exceeds buffer");
    if (!destination.empty())
        detail::download_buffer(impl_, destination.data(), destination.size_bytes());
}
} // namespace gpu
