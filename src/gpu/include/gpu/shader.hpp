#pragma once

#include "types.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>

namespace gpu {
namespace detail { struct ShaderImpl; }
namespace detail { class DeviceImpl; }

class Shader {
public:
    Shader() = default;
    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }
    std::string_view entry_point() const noexcept { return entry_point_; }

private:
    friend class Device;
    friend class detail::DeviceImpl;
    friend class ComputePipeline;
    Shader(std::shared_ptr<detail::ShaderImpl> impl, std::string_view entry_point)
        : impl_(std::move(impl)), entry_point_(entry_point) {}
    std::shared_ptr<detail::ShaderImpl> impl_;
    std::string_view entry_point_ = "main";
};
} // namespace gpu
