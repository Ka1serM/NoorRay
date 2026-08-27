#pragma once

#include "memory.hpp"
#include "shader.hpp"

#include <memory>
#include <type_traits>

namespace gpu {
namespace detail { class ComputePipelineImpl; }

class ComputePipeline {
public:
    ComputePipeline() = default;
    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

    template<class Args>
    void launch(DispatchSize groups, const Args& args) const {
        static_assert(std::is_trivially_copyable_v<Args>, "GPU arguments must be trivially copyable");
        launch_bytes(groups, &args, sizeof(Args));
    }

    // The dispatch dimensions are read from GPU memory; the root arguments
    // are supplied by the host exactly as they are for a direct launch.
    template<class Args>
    void launch_indirect(GpuPtr<DispatchArgs> groups, const Args& args) const {
        static_assert(std::is_trivially_copyable_v<Args>, "GPU arguments must be trivially copyable");
        launch_indirect_bytes(groups, &args, sizeof(Args));
    }

private:
    friend class Device;
    explicit ComputePipeline(std::shared_ptr<detail::ComputePipelineImpl> impl)
        : impl_(std::move(impl)) {}
    void launch_bytes(DispatchSize groups, const void* args, std::size_t size) const;
    void launch_indirect_bytes(GpuPtr<DispatchArgs> groups, const void* args,
        std::size_t size) const;
    std::shared_ptr<detail::ComputePipelineImpl> impl_;
};
} // namespace gpu
