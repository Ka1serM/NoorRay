#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu {

enum class ErrorCode {
    UnsupportedFeature,
    InvalidArgument,
    InvalidShader,
    ShaderCreationFailed,
    OutOfMemory,
    DeviceLost,
    InvalidResource,
    InvalidState,
};

class Error final : public std::runtime_error {
public:
    Error(ErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

template<class T>
struct alignas(8) GpuPtr {
    std::uint64_t address = 0;
};

static_assert(sizeof(GpuPtr<float>) == 8 && alignof(GpuPtr<float>) == 8);

struct float2 { float x{}, y{}; };
struct alignas(16) float3 { float x{}, y{}, z{}; };
struct alignas(16) float4 { float x{}, y{}, z{}, w{}; };
struct int2 { std::int32_t x{}, y{}; };
struct alignas(16) int3 { std::int32_t x{}, y{}, z{}; };
struct alignas(16) int4 { std::int32_t x{}, y{}, z{}, w{}; };
struct uint2 { std::uint32_t x{}, y{}; };
struct alignas(16) uint3 { std::uint32_t x{}, y{}, z{}; };
struct alignas(16) uint4 { std::uint32_t x{}, y{}, z{}, w{}; };

struct float2x2 { float values[2][2]{}; };
struct alignas(16) float3x3 { float values[3][4]{}; };
struct alignas(16) float4x4 { float values[4][4]{}; };

inline float dot(const float2 a, const float2 b) { return a.x * b.x + a.y * b.y; }
inline float dot(const float3 a, const float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float dot(const float4 a, const float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline float3 cross(const float3 a, const float3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(const float2 v) { return std::sqrt(dot(v, v)); }
inline float length(const float3 v) { return std::sqrt(dot(v, v)); }
inline float length(const float4 v) { return std::sqrt(dot(v, v)); }
inline float2 normalize(const float2 v) { const float n = length(v); return n ? float2{v.x / n, v.y / n} : float2{}; }
inline float3 normalize(const float3 v) { const float n = length(v); return n ? float3{v.x / n, v.y / n, v.z / n} : float3{}; }
inline float4 normalize(const float4 v) { const float n = length(v); return n ? float4{v.x / n, v.y / n, v.z / n, v.w / n} : float4{}; }

template<class T> constexpr T min(const T a, const T b) { return std::min(a, b); }
template<class T> constexpr T max(const T a, const T b) { return std::max(a, b); }
template<class T> constexpr T clamp(const T value, const T low, const T high) { return std::clamp(value, low, high); }
template<class T, class U> constexpr T lerp(const T a, const T b, const U t) { return a + (b - a) * t; }

enum class Stage { Copy, Compute, Vertex, Fragment, RayTracing, AccelerationStructure, Present };

struct GpuToken { std::uint64_t value = 0; };

struct DispatchSize { std::uint32_t x = 1, y = 1, z = 1; };
struct DispatchArgs { std::uint32_t x = 1, y = 1, z = 1; };
struct DrawArgs {
    std::uint32_t vertex_count = 0;
    std::uint32_t instance_count = 1;
    std::uint32_t first_vertex = 0;
    std::uint32_t first_instance = 0;
};

// Opaque shader-facing resource token. Ordinary buffers use GpuPtr<T>; this
// token is only needed by shaders that deliberately consume descriptor-backed
// resources such as images or explicit resource tables.
struct ResourceHandle {
    std::uint64_t value = 0;
    explicit operator bool() const noexcept { return value != 0; }
};

struct ImageHandle {
    std::uint64_t value = 0;
    explicit operator bool() const noexcept { return value != 0; }
};

struct SamplerHandle {
    std::uint64_t value = 0;
    explicit operator bool() const noexcept { return value != 0; }
};

struct AccelerationStructureHandle {
    std::uint64_t value = 0;
    explicit operator bool() const noexcept { return value != 0; }
};

} // namespace gpu
