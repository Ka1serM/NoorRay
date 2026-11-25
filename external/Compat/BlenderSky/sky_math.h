//! The vector math that Blender's sky scattering sources include as sky_math.h; NoorRay writes its own, because Blender's header is GPL.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

inline constexpr float M_PI_F = 3.14159265358979323846f;
inline constexpr float M_PI_2_F = 1.57079632679489661923f;
inline constexpr float M_2PI_F = 6.28318530717958647692f;
inline constexpr float M_1_PI_F = 0.318309886183790671538f;
inline constexpr float M_1_4PI_F = 0.0795774715459476678844f;

template <int N> struct SkyVector;

template <> struct SkyVector<2> {
    float x = 0.0f, y = 0.0f;
    float& operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
};

template <> struct SkyVector<3> {
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float& operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
    float length() const { return std::sqrt(x * x + y * y + z * z); }
};

template <> struct SkyVector<4> {
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;
    float& operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
};

using float2 = SkyVector<2>;
using float3 = SkyVector<3>;
using float4 = SkyVector<4>;

inline float2 make_float2(float x, float y) { return {x, y}; }
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }

template <int N, typename F> SkyVector<N> skyMap(const SkyVector<N>& a, const SkyVector<N>& b, F f) {
    SkyVector<N> r;
    for (int i = 0; i < N; ++i)
        r[i] = f(a[i], b[i]);
    return r;
}

template <int N> SkyVector<N> skySplat(float s) {
    SkyVector<N> r;
    for (int i = 0; i < N; ++i)
        r[i] = s;
    return r;
}

template <int N> SkyVector<N> operator+(const SkyVector<N>& a, const SkyVector<N>& b) { return skyMap(a, b, [](float p, float q) { return p + q; }); }
template <int N> SkyVector<N> operator-(const SkyVector<N>& a, const SkyVector<N>& b) { return skyMap(a, b, [](float p, float q) { return p - q; }); }
template <int N> SkyVector<N> operator*(const SkyVector<N>& a, const SkyVector<N>& b) { return skyMap(a, b, [](float p, float q) { return p * q; }); }
template <int N> SkyVector<N> operator/(const SkyVector<N>& a, const SkyVector<N>& b) { return skyMap(a, b, [](float p, float q) { return p / q; }); }
template <int N> SkyVector<N> operator*(const SkyVector<N>& a, float s) { return a * skySplat<N>(s); }
template <int N> SkyVector<N> operator*(float s, const SkyVector<N>& a) { return a * skySplat<N>(s); }
template <int N> SkyVector<N> operator/(const SkyVector<N>& a, float s) { return a / skySplat<N>(s); }
template <int N> SkyVector<N> operator-(const SkyVector<N>& a) { return a * -1.0f; }
template <int N> SkyVector<N>& operator+=(SkyVector<N>& a, const SkyVector<N>& b) { return a = a + b; }
template <int N> SkyVector<N>& operator-=(SkyVector<N>& a, const SkyVector<N>& b) { return a = a - b; }
template <int N> SkyVector<N>& operator*=(SkyVector<N>& a, const SkyVector<N>& b) { return a = a * b; }
template <int N> SkyVector<N>& operator*=(SkyVector<N>& a, float s) { return a = a * s; }
template <int N> SkyVector<N>& operator/=(SkyVector<N>& a, float s) { return a = a / s; }

inline float sqr(float a) { return a * a; }
inline float safe_sqrtf(float a) { return std::sqrt(std::max(a, 0.0f)); }
inline float saturate(float a) { return std::clamp(a, 0.0f, 1.0f); }
inline float clamp(float a, float low, float high) { return std::clamp(a, low, high); }
inline float mix(float a, float b, float t) { return a + (b - a) * t; }

template <int N> float reduce_add(const SkyVector<N>& a) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i)
        sum += a[i];
    return sum;
}
template <int N> float dot(const SkyVector<N>& a, const SkyVector<N>& b) { return reduce_add(a * b); }
template <int N> float len_squared(const SkyVector<N>& a) { return dot(a, a); }
template <int N> float len(const SkyVector<N>& a) { return std::sqrt(len_squared(a)); }
template <int N> float distance(const SkyVector<N>& a, const SkyVector<N>& b) { return len(a - b); }
template <int N> SkyVector<N> mix(const SkyVector<N>& a, const SkyVector<N>& b, float t) { return a + (b - a) * t; }
template <int N> SkyVector<N> exp(const SkyVector<N>& a) { return skyMap(a, a, [](float p, float) { return std::exp(p); }); }
template <int N> SkyVector<N> max(const SkyVector<N>& a, float s) { return skyMap(a, skySplat<N>(s), [](float p, float q) { return std::max(p, q); }); }

inline float3 sun_direction(float cos_theta) { return make_float3(-std::sqrt(std::max(1.0f - cos_theta * cos_theta, 0.0f)), 0.0f, cos_theta); }

//! The distance along a unit direction from a point to a sphere at the origin, or -1 when the ray misses it.
inline float ray_sphere_intersection(const float3& origin, const float3& direction, float radius) {
    const float b = dot(origin, direction);
    const float c = dot(origin, origin) - radius * radius;
    if (c > 0.0f && b > 0.0f)
        return -1.0f;
    const float d = b * b - c;
    if (d < 0.0f)
        return -1.0f;
    if (d >= b * b)
        return -b + std::sqrt(d);
    return -b - std::sqrt(d);
}

template <typename F> void SKY_parallel_for(std::size_t begin, std::size_t end, std::size_t grain, const F& function) {
    const std::size_t count = std::max<std::size_t>(std::thread::hardware_concurrency(), 1);
    const std::size_t step = std::max(grain, (end - begin + count - 1) / count);
    std::vector<std::jthread> workers;
    for (std::size_t first = begin; first < end; first += step)
        workers.emplace_back([&function, first, last = std::min(first + step, end)] { function(first, last); });
}
