#pragma once

#include "types.hpp"

#include <memory>

namespace gpu {

enum class Filter { Nearest, Linear };
enum class AddressMode { Repeat, MirroredRepeat, ClampToEdge, ClampToBorder };
struct SamplerDesc {
    Filter filter = Filter::Linear;
    AddressMode address_u = AddressMode::Repeat;
    AddressMode address_v = AddressMode::Repeat;
    AddressMode address_w = AddressMode::Repeat;
};

namespace detail {
struct SamplerImpl;
std::uint32_t sampler_handle(const std::shared_ptr<SamplerImpl>&);
}

class Sampler {
public:
    Sampler() = default;
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&&) noexcept = default;
    Sampler& operator=(Sampler&&) noexcept = default;
    // Sampler-heap index; shaders read it as SamplerDescriptorHeap[i].
    SamplerHandle handle() const noexcept {
        return impl_ ? SamplerHandle{detail::sampler_handle(impl_)} : SamplerHandle{};
    }
    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
    friend class Device;
    explicit Sampler(std::shared_ptr<detail::SamplerImpl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<detail::SamplerImpl> impl_;
};
} // namespace gpu
