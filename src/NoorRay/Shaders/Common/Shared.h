#pragma once

//! C++ and Slang both include this. Shared bodies use NR_SHARED, f-suffixed float literals, no swizzles and no broadcast constructors.
#if defined(__SLANG__)
#define NR_SHARED
#define NR_NAMESPACE_BEGIN
#define NR_NAMESPACE_END
#else
#include <algorithm>
#include <cmath>

#include "NoorRay/Core/Math.h"
#define NR_SHARED inline
#define NR_NAMESPACE_BEGIN                                                                                                                                     \
    namespace NoorRay {                                                                                                                                        \
    namespace Shaders {                                                                                                                                        \
    using namespace NoorRHI;
#define NR_NAMESPACE_END                                                                                                                                       \
    }                                                                                                                                                          \
    }
namespace NoorRay {
namespace Shaders {
using Core::cross;
using Core::dot;
using Core::float2;
using Core::float3;
using Core::float4;
using Core::length;
using NoorRHI::uint;
using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::ceil;
using std::cos;
using std::cosh;
using std::exp;
using std::floor;
using std::fmod;
using std::log;
using std::pow;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;
inline float min(const float a, const float b) {
    return std::min(a, b);
}
inline float max(const float a, const float b) {
    return std::max(a, b);
}
inline float clamp(const float x, const float low, const float high) {
    return std::min(std::max(x, low), high);
}
inline float saturate(const float x) {
    return clamp(x, 0.0f, 1.0f);
}
inline float lerp(const float a, const float b, const float t) {
    return a + (b - a) * t;
}
inline float3 lerp(const float3 a, const float3 b, const float t) {
    return a + (b - a) * t;
}
inline float smoothstep(const float low, const float high, const float x) {
    const float t = saturate((x - low) / (high - low));
    return t * t * (3.0f - 2.0f * t);
}
}
}
#endif
