#pragma once

#include "image.hpp"
#include "shader.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace gpu {

class RayTracingPipeline;

namespace detail {
class DeviceImpl;
class RayTracingPipelineImpl;
struct AccelerationStructureImpl;
}

struct TriangleGeometry {
    GpuPtr<float3> positions;
    GpuPtr<std::uint32_t> indices;
    std::uint32_t triangle_count = 0;
    // Opaque geometry skips any-hit invocation. Gaussian proxy triangles set
    // this to false so their stochastic acceptance shader can reject hits.
    bool opaque = true;
};

class AccelerationStructure {
public:
    AccelerationStructure() = default;
    AccelerationStructureHandle handle() const noexcept;
    // Device address of the structure itself, for writing into an
    // InstanceRecord from a shader.
    std::uint64_t device_address() const;
    explicit operator bool() const noexcept;

private:
    friend class Device;
    friend class detail::DeviceImpl;
    explicit AccelerationStructure(std::shared_ptr<detail::AccelerationStructureImpl> impl)
        : impl_(std::move(impl)) {}
    std::shared_ptr<detail::AccelerationStructureImpl> impl_;
};

// Bit-exact mirror of VkAccelerationStructureInstanceKHR. Exposing the packed
// record lets a compute shader produce TLAS input directly in device-local
// memory, which avoids staging one 64-byte record per instance through the
// host. Scenes with millions of instances cannot afford that staging buffer:
// it is host-visible, and on a system without resizable BAR the host-visible
// device-local heap is only a few hundred megabytes.
struct InstanceRecord {
    // Row-major 3x4 object-to-world affine transform.
    float transform[12];
    // Low 24 bits the custom index, high 8 bits the visibility mask.
    std::uint32_t custom_index_and_mask;
    // Low 24 bits the hit-group record offset, high 8 bits the instance flags.
    std::uint32_t shader_binding_table_offset_and_flags;
    std::uint64_t blas_address;
};
static_assert(sizeof(InstanceRecord) == 64);

// Matches VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR, which is
// what the host-span overloads below apply to every instance they build.
inline constexpr std::uint32_t InstanceFlagTriangleFacingCullDisable = 0x1u;

struct Instance {
    AccelerationStructure blas{};
    float4x4 transform{};
    // Vulkan packs these fields into the TLAS instance record. Keeping them
    // in the clean API lets callers select hit groups without touching Vulkan.
    std::uint32_t custom_index = 0;
    std::uint32_t shader_binding_table_offset = 0;
    std::uint8_t mask = 0xff;
};
struct RayTracingPipelineDesc {
    Shader raygen;
    std::vector<Shader> miss;
    std::vector<Shader> closest_hit;
    std::vector<Shader> any_hit;
    std::vector<Shader> intersection;
};

class RayTracingPipeline {
public:
    RayTracingPipeline() = default;
    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }
    template<class Args>
    void trace(DispatchSize size, const Args& args) const {
        static_assert(std::is_trivially_copyable_v<Args>, "GPU arguments must be trivially copyable");
        trace_bytes(size, &args, sizeof(Args));
    }

private:
    friend class Device;
    explicit RayTracingPipeline(std::shared_ptr<detail::RayTracingPipelineImpl> impl)
        : impl_(std::move(impl)) {}
    void trace_bytes(DispatchSize, const void*, std::size_t) const;
    std::shared_ptr<detail::RayTracingPipelineImpl> impl_;
};
} // namespace gpu
