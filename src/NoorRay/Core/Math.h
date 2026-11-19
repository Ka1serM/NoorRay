#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>

#include "NoorRHI/Core/Types.h"

//! The operators live in the namespace of the NoorRHI vector types, so that argument-dependent lookup finds them everywhere.
namespace NoorRHI {

constexpr float2 operator+(const float2 a, const float2 b) {
    return {a.x + b.x, a.y + b.y};
}
constexpr float2 operator-(const float2 a, const float2 b) {
    return {a.x - b.x, a.y - b.y};
}
constexpr float2 operator*(const float2 a, const float s) {
    return {a.x * s, a.y * s};
}

constexpr float3 operator+(const float3 a, const float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
constexpr float3 operator-(const float3 a, const float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
constexpr float3 operator-(const float3 a) {
    return {-a.x, -a.y, -a.z};
}
constexpr float3 operator*(const float3 a, const float3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}
constexpr float3 operator*(const float3 a, const float s) {
    return {a.x * s, a.y * s, a.z * s};
}
constexpr float3 operator*(const float s, const float3 a) {
    return a * s;
}
constexpr float3 operator/(const float3 a, const float s) {
    return {a.x / s, a.y / s, a.z / s};
}
constexpr float3& operator+=(float3& a, const float3 b) {
    return a = a + b;
}
constexpr bool operator==(const float3 a, const float3 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

constexpr float4 operator+(const float4 a, const float4 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
constexpr float4 operator*(const float4 a, const float s) {
    return {a.x * s, a.y * s, a.z * s, a.w * s};
}
constexpr bool operator==(const float4 a, const float4 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

}

namespace NoorRay::Core {

using NoorRHI::float2;
using NoorRHI::float3;
using NoorRHI::float4;
using NoorRHI::float4x4;
using NoorRHI::half;
using NoorRHI::uint2;
using NoorRHI::uint3;
using NoorRHI::uint4;

inline constexpr float pi = std::numbers::pi_v<float>;

constexpr float dot(const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
constexpr float3 cross(const float3 a, const float3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(const float3 a) {
    return std::sqrt(dot(a, a));
}
inline float3 normalize(const float3 a) {
    const float size = length(a);
    return size > 0.0f ? a / size : float3{};
}
constexpr float3 min(const float3 a, const float3 b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}
constexpr float3 max(const float3 a, const float3 b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}
constexpr float maxComponent(const float3 a) {
    return std::max({a.x, a.y, a.z});
}
constexpr float luminance(const float3 rgb) {
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

struct Quaternion {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

struct Bounds {
    float3 min{INFINITY, INFINITY, INFINITY};
    float3 max{-INFINITY, -INFINITY, -INFINITY};

    constexpr void grow(const float3 point) {
        min = Core::min(min, point);
        max = Core::max(max, point);
    }
    constexpr void grow(const Bounds& other) {
        min = Core::min(min, other.min);
        max = Core::max(max, other.max);
    }
    constexpr bool isEmpty() const { return min.x > max.x; }
    constexpr float3 center() const { return (min + max) * 0.5f; }
    constexpr float3 extent() const { return max - min; }
};

//! Row-major 3x4 affine transform, the layout of a TLAS instance and of the scene file.
struct Transform {
    float rows[3][4]{{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}};

    static Transform fromTrs(float3 translation, Quaternion rotation, float3 scale);
    static Transform lookAt(float3 eye, float3 target, float3 up);

    constexpr float3 getColumn(const int column) const { return {rows[0][column], rows[1][column], rows[2][column]}; }
    constexpr float3 getTranslation() const { return getColumn(3); }
    constexpr float3 transformPoint(const float3 p) const {
        return {rows[0][0] * p.x + rows[0][1] * p.y + rows[0][2] * p.z + rows[0][3], rows[1][0] * p.x + rows[1][1] * p.y + rows[1][2] * p.z + rows[1][3],
                rows[2][0] * p.x + rows[2][1] * p.y + rows[2][2] * p.z + rows[2][3]};
    }
    constexpr float3 transformVector(const float3 v) const {
        return {rows[0][0] * v.x + rows[0][1] * v.y + rows[0][2] * v.z, rows[1][0] * v.x + rows[1][1] * v.y + rows[1][2] * v.z,
                rows[2][0] * v.x + rows[2][1] * v.y + rows[2][2] * v.z};
    }
    Transform operator*(const Transform& other) const;
    Transform inverse() const;
    float determinant() const;
    Bounds transformBounds(const Bounds& bounds) const;
    float4x4 toMatrix() const;
    bool operator==(const Transform&) const = default;
};

float4x4 multiply(const float4x4& a, const float4x4& b);
float4x4 inverse(const float4x4& m);
float4x4 perspective(float verticalFov, float aspect, float near, float far);
float4x4 orthographic(float height, float aspect, float near, float far);

constexpr std::uint32_t floatToBits(const float value) {
    return std::bit_cast<std::uint32_t>(value);
}
half toHalf(float value);
float fromHalf(half value);
std::uint32_t packHalf2(float a, float b);
//! Octahedral unit vector in two signed 16-bit components.
std::uint32_t packOctahedral(float3 direction);
float3 unpackOctahedral(std::uint32_t packed);
//! Octahedral tangent: u as snorm16, v as snorm15, and the bitangent sign in bit 31.
std::uint32_t packTangent(float3 tangent, float bitangentSign);
std::uint32_t packUnorm4x8(float4 value);

constexpr std::uint32_t divideRoundUp(const std::uint32_t value, const std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

}
