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

namespace detail { struct SamplerImpl; }

class Sampler {
public:
    Sampler() = default;
    SamplerHandle handle() const noexcept;
    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
    friend class Device;
    explicit Sampler(std::shared_ptr<detail::SamplerImpl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<detail::SamplerImpl> impl_;
};
} // namespace gpu
