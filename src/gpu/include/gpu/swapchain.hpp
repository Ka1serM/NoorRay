#pragma once

#include "image.hpp"
#include "types.hpp"

#include <cstdint>
#include <memory>

namespace gpu {

namespace detail { class SwapchainImpl; class DeviceImpl; }
namespace interop { std::uintptr_t command_buffer(const class Frame&); }

// A presentation chain owned by the library. Swapchain images are ordinary
// ImageHandles from the caller's point of view: they go into a RenderTarget,
// they are read by copy_to, and the layout transitions presentation needs are
// the library's business, not the caller's.
//
// The chain rebuilds itself whenever the surface size stops matching the
// provider's, so an application never handles VK_ERROR_OUT_OF_DATE_KHR.
class Swapchain {
public:
    Swapchain() = default;

    ImageFormat format() const noexcept;
    std::uint32_t width() const noexcept;
    std::uint32_t height() const noexcept;
    std::uint32_t image_count() const noexcept;

    // Force a rebuild before the next acquire. Applications normally do not
    // need this - a size mismatch is detected on its own - but a window
    // manager can resize a surface without changing the reported size, and a
    // resize event is a cheap way to stay ahead of it.
    void invalidate();

    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
    friend class Device;
    friend class Frame;
    friend class detail::DeviceImpl;
    explicit Swapchain(std::shared_ptr<detail::SwapchainImpl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<detail::SwapchainImpl> impl_;
};

// One acquired presentation image, plus the command buffer that every
// operation recorded between begin_frame and end_frame goes into. Dispatches,
// traces and render scopes issued while a frame is open are batched into that
// one submission instead of being submitted individually.
//
// A falsy Frame means no image was acquired because the chain was rebuilt;
// skip the frame and call begin_frame again. Frames are move-only, and
// dropping one without end_frame discards its recorded work.
class Frame {
public:
    Frame() = default;
    Frame(Frame&&) noexcept;
    Frame& operator=(Frame&&) noexcept;
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
    ~Frame();

    // The acquired image, usable as a RenderTarget colour attachment.
    ImageHandle target() const noexcept;
    std::uint32_t index() const noexcept;
    std::uint32_t width() const noexcept;
    std::uint32_t height() const noexcept;
    ImageFormat format() const noexcept;

    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
    friend class Device;
    friend class detail::DeviceImpl;
    friend std::uintptr_t interop::command_buffer(const Frame&);
    struct State;
    explicit Frame(std::shared_ptr<State> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<State> impl_;
};

} // namespace gpu
