#pragma once

#include "device.hpp"

#include <cstdint>
#include <memory>

namespace gpu {

// The small format vocabulary keeps image/pipeline compatibility explicit
// without exposing backend-specific format enums in the public API.
enum class ImageFormat {
    Auto,
    Rgba8Unorm,
    Bgra8Unorm,
    Rgba32Float,
    R32Uint,
    D32Float,
};

enum class ImageUsage : std::uint32_t {
    Sampled = 1u << 0,
    Storage = 1u << 1,
    ColorAttachment = 1u << 2,
    DepthAttachment = 1u << 3,
    TransferSource = 1u << 4,
    TransferDestination = 1u << 5,
    // Allocate dedicated Vulkan memory that can be exported through
    // gpu::interop for another Vulkan-capable API, such as OpenGL's
    // GL_EXT_memory_object_fd implementation. This is deliberately opt-in:
    // ordinary images keep using VMA's compact internal allocations.
    ExternalMemory = 1u << 6,
};
constexpr ImageUsage operator|(ImageUsage a, ImageUsage b) {
    return static_cast<ImageUsage>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

namespace detail { struct ImageImpl; }

template<class T>
class Image {
public:
    Image() = default;
    ImageHandle handle() const noexcept;
    ImageHandle sampled_handle() const noexcept;
    ImageHandle storage_handle() const noexcept;
    ImageFormat format() const noexcept { return format_; }
    std::uint32_t width() const noexcept { return width_; }
    std::uint32_t height() const noexcept { return height_; }
    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
    friend class Device;
    template<class U> friend class Image;
    Image(std::shared_ptr<detail::ImageImpl> impl, std::uint32_t width, std::uint32_t height,
          ImageFormat format)
        : impl_(std::move(impl)), width_(width), height_(height), format_(format) {}
    std::shared_ptr<detail::ImageImpl> impl_;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    ImageFormat format_ = ImageFormat::Auto;
};

} // namespace gpu

namespace gpu::detail {
std::shared_ptr<ImageImpl> make_image(const std::shared_ptr<DeviceImpl>&, std::uint32_t,
                                      std::uint32_t, ImageUsage, ImageFormat);
std::uint64_t image_handle(const std::shared_ptr<ImageImpl>&);
std::uint64_t image_sampled_handle(const std::shared_ptr<ImageImpl>&);
std::uint64_t image_storage_handle(const std::shared_ptr<ImageImpl>&);
void upload_image(const std::shared_ptr<DeviceImpl>&, const std::shared_ptr<ImageImpl>&,
                  const void*, std::size_t);
void download_image(const std::shared_ptr<DeviceImpl>&, const std::shared_ptr<ImageImpl>&,
                    void*, std::size_t);
std::size_t image_byte_size(const std::shared_ptr<ImageImpl>&);
}

namespace gpu {
template<class T>
Image<T> Device::image(const std::uint32_t width, const std::uint32_t height,
    const ImageUsage usage) {
    return image<T>(width, height, usage, ImageFormat::Auto);
}

template<class T>
Image<T> Device::image(const std::uint32_t width, const std::uint32_t height,
    const ImageUsage usage, const ImageFormat format) {
    if (width == 0 || height == 0)
        throw Error(ErrorCode::InvalidArgument, "gpu::Device::image requires non-zero dimensions");
    const ImageFormat actual_format = format == ImageFormat::Auto
        ? ((static_cast<std::uint32_t>(usage)
            & static_cast<std::uint32_t>(ImageUsage::DepthAttachment))
            ? ImageFormat::D32Float : ImageFormat::Rgba8Unorm)
        : format;
    return Image<T>(detail::make_image(impl_, width, height, usage, actual_format),
        width, height, actual_format);
}

template<class T>
ImageHandle Image<T>::handle() const noexcept {
    return impl_ ? ImageHandle{detail::image_handle(impl_)} : ImageHandle{};
}

template<class T>
ImageHandle Image<T>::sampled_handle() const noexcept {
    return impl_ ? ImageHandle{detail::image_sampled_handle(impl_)} : ImageHandle{};
}

template<class T>
ImageHandle Image<T>::storage_handle() const noexcept {
    return impl_ ? ImageHandle{detail::image_storage_handle(impl_)} : ImageHandle{};
}

template<class T>
void Device::upload(Image<T>& destination, const std::span<const T> data) {
    if (!destination.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot upload to an empty gpu::Image");
    if (data.size_bytes() != detail::image_byte_size(destination.impl_))
        throw Error(ErrorCode::InvalidArgument,
            "image upload must cover exactly one full mip level");
    detail::upload_image(impl_, destination.impl_, data.data(), data.size_bytes());
}

template<class T>
void Device::download(const std::span<T> destination, const Image<T>& source) {
    if (!source.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot download from an empty gpu::Image");
    if (destination.size_bytes() != detail::image_byte_size(source.impl_))
        throw Error(ErrorCode::InvalidArgument,
            "image download must cover exactly one full mip level");
    detail::download_image(impl_, source.impl_, destination.data(), destination.size_bytes());
}
} // namespace gpu
