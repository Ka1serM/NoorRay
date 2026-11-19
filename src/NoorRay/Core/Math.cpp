#include "NoorRay/Core/Math.h"

namespace NoorRay::Core {

Transform Transform::fromTrs(const float3 translation, const Quaternion q, const float3 scale) {
    const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    Transform result;
    result.rows[0][0] = (1.0f - 2.0f * (yy + zz)) * scale.x;
    result.rows[0][1] = 2.0f * (xy - wz) * scale.y;
    result.rows[0][2] = 2.0f * (xz + wy) * scale.z;
    result.rows[1][0] = 2.0f * (xy + wz) * scale.x;
    result.rows[1][1] = (1.0f - 2.0f * (xx + zz)) * scale.y;
    result.rows[1][2] = 2.0f * (yz - wx) * scale.z;
    result.rows[2][0] = 2.0f * (xz - wy) * scale.x;
    result.rows[2][1] = 2.0f * (yz + wx) * scale.y;
    result.rows[2][2] = (1.0f - 2.0f * (xx + yy)) * scale.z;
    result.rows[0][3] = translation.x;
    result.rows[1][3] = translation.y;
    result.rows[2][3] = translation.z;
    return result;
}

Transform Transform::lookAt(const float3 eye, const float3 target, const float3 up) {
    const float3 back = normalize(eye - target);
    const float3 right = normalize(cross(up, back));
    const float3 trueUp = cross(back, right);
    Transform result;
    const float3 columns[] = {right, trueUp, back, eye};
    for (int column = 0; column < 4; ++column) {
        result.rows[0][column] = columns[column].x;
        result.rows[1][column] = columns[column].y;
        result.rows[2][column] = columns[column].z;
    }
    return result;
}

Transform Transform::operator*(const Transform& other) const {
    Transform result;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 4; ++column) {
            float value = column == 3 ? rows[row][3] : 0.0f;
            for (int k = 0; k < 3; ++k)
                value += rows[row][k] * other.rows[k][column];
            result.rows[row][column] = value;
        }
    }
    return result;
}

float Transform::determinant() const {
    return dot(getColumn(0), cross(getColumn(1), getColumn(2)));
}

Transform Transform::inverse() const {
    const float3 c0 = getColumn(0), c1 = getColumn(1), c2 = getColumn(2);
    const float3 r0 = cross(c1, c2), r1 = cross(c2, c0), r2 = cross(c0, c1);
    const float inverseDeterminant = 1.0f / dot(c0, r0);
    Transform result;
    const float3 inverseRows[] = {r0 * inverseDeterminant, r1 * inverseDeterminant, r2 * inverseDeterminant};
    const float3 translation = getTranslation();
    for (int row = 0; row < 3; ++row) {
        result.rows[row][0] = inverseRows[row].x;
        result.rows[row][1] = inverseRows[row].y;
        result.rows[row][2] = inverseRows[row].z;
        result.rows[row][3] = -dot(inverseRows[row], translation);
    }
    return result;
}

Bounds Transform::transformBounds(const Bounds& bounds) const {
    Bounds result;
    if (bounds.isEmpty())
        return result;
    for (int corner = 0; corner < 8; ++corner) {
        const float3 point{corner & 1 ? bounds.max.x : bounds.min.x, corner & 2 ? bounds.max.y : bounds.min.y, corner & 4 ? bounds.max.z : bounds.min.z};
        result.grow(transformPoint(point));
    }
    return result;
}

float4x4 Transform::toMatrix() const {
    float4x4 result;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 4; ++column)
            result.values[row][column] = rows[row][column];
    result.values[3][3] = 1.0f;
    return result;
}

float4x4 multiply(const float4x4& a, const float4x4& b) {
    float4x4 result;
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            for (int k = 0; k < 4; ++k)
                result.values[row][column] += a.values[row][k] * b.values[k][column];
    return result;
}

float4x4 inverse(const float4x4& m) {
    const auto& a = m.values;
    float s[6], c[6];
    s[0] = a[0][0] * a[1][1] - a[1][0] * a[0][1];
    s[1] = a[0][0] * a[1][2] - a[1][0] * a[0][2];
    s[2] = a[0][0] * a[1][3] - a[1][0] * a[0][3];
    s[3] = a[0][1] * a[1][2] - a[1][1] * a[0][2];
    s[4] = a[0][1] * a[1][3] - a[1][1] * a[0][3];
    s[5] = a[0][2] * a[1][3] - a[1][2] * a[0][3];
    c[5] = a[2][2] * a[3][3] - a[3][2] * a[2][3];
    c[4] = a[2][1] * a[3][3] - a[3][1] * a[2][3];
    c[3] = a[2][1] * a[3][2] - a[3][1] * a[2][2];
    c[2] = a[2][0] * a[3][3] - a[3][0] * a[2][3];
    c[1] = a[2][0] * a[3][2] - a[3][0] * a[2][2];
    c[0] = a[2][0] * a[3][1] - a[3][0] * a[2][1];
    const float inverseDeterminant = 1.0f / (s[0] * c[5] - s[1] * c[4] + s[2] * c[3] + s[3] * c[2] - s[4] * c[1] + s[5] * c[0]);
    float4x4 result;
    auto& b = result.values;
    b[0][0] = (a[1][1] * c[5] - a[1][2] * c[4] + a[1][3] * c[3]) * inverseDeterminant;
    b[0][1] = (-a[0][1] * c[5] + a[0][2] * c[4] - a[0][3] * c[3]) * inverseDeterminant;
    b[0][2] = (a[3][1] * s[5] - a[3][2] * s[4] + a[3][3] * s[3]) * inverseDeterminant;
    b[0][3] = (-a[2][1] * s[5] + a[2][2] * s[4] - a[2][3] * s[3]) * inverseDeterminant;
    b[1][0] = (-a[1][0] * c[5] + a[1][2] * c[2] - a[1][3] * c[1]) * inverseDeterminant;
    b[1][1] = (a[0][0] * c[5] - a[0][2] * c[2] + a[0][3] * c[1]) * inverseDeterminant;
    b[1][2] = (-a[3][0] * s[5] + a[3][2] * s[2] - a[3][3] * s[1]) * inverseDeterminant;
    b[1][3] = (a[2][0] * s[5] - a[2][2] * s[2] + a[2][3] * s[1]) * inverseDeterminant;
    b[2][0] = (a[1][0] * c[4] - a[1][1] * c[2] + a[1][3] * c[0]) * inverseDeterminant;
    b[2][1] = (-a[0][0] * c[4] + a[0][1] * c[2] - a[0][3] * c[0]) * inverseDeterminant;
    b[2][2] = (a[3][0] * s[4] - a[3][1] * s[2] + a[3][3] * s[0]) * inverseDeterminant;
    b[2][3] = (-a[2][0] * s[4] + a[2][1] * s[2] - a[2][3] * s[0]) * inverseDeterminant;
    b[3][0] = (-a[1][0] * c[3] + a[1][1] * c[1] - a[1][2] * c[0]) * inverseDeterminant;
    b[3][1] = (a[0][0] * c[3] - a[0][1] * c[1] + a[0][2] * c[0]) * inverseDeterminant;
    b[3][2] = (-a[3][0] * s[3] + a[3][1] * s[1] - a[3][2] * s[0]) * inverseDeterminant;
    b[3][3] = (a[2][0] * s[3] - a[2][1] * s[1] + a[2][2] * s[0]) * inverseDeterminant;
    return result;
}

float4x4 perspective(const float verticalFov, const float aspect, const float near, const float far) {
    const float focal = 1.0f / std::tan(verticalFov * 0.5f);
    float4x4 result;
    result.values[0][0] = focal / aspect;
    result.values[1][1] = -focal;
    result.values[2][2] = near / (far - near);
    result.values[2][3] = far * near / (far - near);
    result.values[3][2] = -1.0f;
    return result;
}

float4x4 orthographic(const float height, const float aspect, const float near, const float far) {
    float4x4 result;
    result.values[0][0] = 2.0f / (height * aspect);
    result.values[1][1] = -2.0f / height;
    result.values[2][2] = 1.0f / (far - near);
    result.values[2][3] = far / (far - near);
    result.values[3][3] = 1.0f;
    return result;
}

//! Rounds to nearest even, like the hardware conversion; wasm32 has no _Float16.
half toHalf(const float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>(bits >> 16 & 0x8000u);
    const std::uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u)
        return half{static_cast<std::uint16_t>(sign | (magnitude > 0x7f800000u ? 0x7e00u : 0x7c00u))};
    if (magnitude >= 0x477ff000u)
        return half{static_cast<std::uint16_t>(sign | 0x7c00u)};
    if (magnitude < 0x38800000u) {
        const float subnormal = std::bit_cast<float>(magnitude) + 0.5f;
        return half{static_cast<std::uint16_t>(sign | (std::bit_cast<std::uint32_t>(subnormal) - 0x3f000000u))};
    }
    const std::uint32_t odd = magnitude >> 13 & 1u;
    return half{static_cast<std::uint16_t>(sign | (magnitude - 0x38000000u + 0xfffu + odd) >> 13)};
}

float fromHalf(const half value) {
    const std::uint32_t sign = std::uint32_t{value.bits & 0x8000u} << 16;
    const std::uint32_t exponent = value.bits >> 10 & 0x1fu;
    const std::uint32_t mantissa = value.bits & 0x3ffu;
    if (exponent == 0x1fu)
        return std::bit_cast<float>(sign | 0x7f800000u | mantissa << 13);
    if (exponent == 0)
        return std::bit_cast<float>(sign | std::bit_cast<std::uint32_t>(static_cast<float>(mantissa) * 5.9604645e-8f));
    return std::bit_cast<float>(sign | (exponent + 112u) << 23 | mantissa << 13);
}

std::uint32_t packHalf2(const float a, const float b) {
    return std::uint32_t{toHalf(a).bits} | std::uint32_t{toHalf(b).bits} << 16;
}

std::uint32_t packOctahedral(const float3 direction) {
    const float3 n = direction / (std::abs(direction.x) + std::abs(direction.y) + std::abs(direction.z));
    float u = n.x, v = n.y;
    if (n.z < 0.0f) {
        u = (1.0f - std::abs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f);
        v = (1.0f - std::abs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f);
    }
    const auto snorm = [](const float x) {
        return static_cast<std::uint32_t>(static_cast<std::uint16_t>(static_cast<std::int16_t>(std::lround(std::clamp(x, -1.0f, 1.0f) * 32767.0f))));
    };
    return snorm(u) | snorm(v) << 16;
}

float3 unpackOctahedral(const std::uint32_t packed) {
    const float u = std::max(static_cast<std::int16_t>(packed & 0xffff) / 32767.0f, -1.0f);
    const float v = std::max(static_cast<std::int16_t>(packed >> 16) / 32767.0f, -1.0f);
    float3 n{u, v, 1.0f - std::abs(u) - std::abs(v)};
    const float t = std::max(-n.z, 0.0f);
    n.x += n.x >= 0.0f ? -t : t;
    n.y += n.y >= 0.0f ? -t : t;
    return normalize(n);
}

std::uint32_t packTangent(const float3 tangent, const float bitangentSign) {
    const std::uint32_t octahedral = packOctahedral(tangent);
    const std::int16_t v = static_cast<std::int16_t>(octahedral >> 16);
    const std::uint32_t v15 = static_cast<std::uint32_t>(v >> 1) & 0x7fffu;
    return (octahedral & 0xffffu) | v15 << 16 | (bitangentSign < 0.0f ? 0x80000000u : 0u);
}

std::uint32_t packUnorm4x8(const float4 value) {
    const auto unorm = [](const float x) { return static_cast<std::uint32_t>(std::lround(std::clamp(x, 0.0f, 1.0f) * 255.0f)); };
    return unorm(value.x) | unorm(value.y) << 8 | unorm(value.z) << 16 | unorm(value.w) << 24;
}

}
