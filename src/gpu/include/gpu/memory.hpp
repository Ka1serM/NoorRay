#pragma once

#include "device.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace gpu {

namespace detail { struct BufferImpl; }

template<class T>
class Buffer {
public:
    Buffer() = default;

    GpuPtr<T> ptr() const;
    ResourceHandle handle() const;
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
void upload_buffer(const std::shared_ptr<DeviceImpl>&, const std::shared_ptr<BufferImpl>&,
                   const void*, std::size_t, std::size_t, std::size_t);
void download_buffer(const std::shared_ptr<DeviceImpl>&, const std::shared_ptr<BufferImpl>&,
                     void*, std::size_t, std::size_t);
std::uint64_t buffer_address(const std::shared_ptr<BufferImpl>&);
ResourceHandle buffer_handle(const std::shared_ptr<BufferImpl>&);
}

namespace gpu {
template<class T>
GpuPtr<T> Buffer<T>::ptr() const {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "gpu::Buffer is empty");
    return {detail::buffer_address(impl_)};
}

template<class T>
ResourceHandle Buffer<T>::handle() const {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "gpu::Buffer is empty");
    return detail::buffer_handle(impl_);
}

template<class T>
Buffer<T> Device::buffer(const std::size_t count) {
    if (count == 0)
        throw Error(ErrorCode::InvalidArgument, "gpu::Device::buffer requires a non-zero count");
    return Buffer<T>(detail::make_buffer(impl_, count * sizeof(T), alignof(T)), count);
}

template<class T>
void Device::upload(Buffer<T>& destination, const std::span<const T> data) {
    if (!destination.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot upload to an empty gpu::Buffer");
    if (data.size() > destination.size())
        throw Error(ErrorCode::InvalidArgument, "upload data is larger than the destination buffer");
    if (data.empty())
        return;
    detail::upload_buffer(impl_, destination.impl_, data.data(), data.size_bytes(), sizeof(T), 0);
}

template<class T>
void Device::upload(Buffer<T>& destination, const std::span<const T> data,
    const std::size_t destination_offset) {
    if (!destination.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot upload to an empty gpu::Buffer");
    if (destination_offset > destination.size()
        || data.size() > destination.size() - destination_offset)
        throw Error(ErrorCode::InvalidArgument, "upload range exceeds the destination buffer");
    if (data.empty())
        return;
    detail::upload_buffer(impl_, destination.impl_, data.data(), data.size_bytes(),
        sizeof(T), destination_offset * sizeof(T));
}

template<class T>
void Device::download(const std::span<T> destination, const Buffer<T>& source) {
    if (!source.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot download from an empty gpu::Buffer");
    if (destination.size() > source.size())
        throw Error(ErrorCode::InvalidArgument, "download destination is larger than the source buffer");
    if (destination.empty())
        return;
    detail::download_buffer(impl_, source.impl_, destination.data(), destination.size_bytes(), sizeof(T));
}
} // namespace gpu
