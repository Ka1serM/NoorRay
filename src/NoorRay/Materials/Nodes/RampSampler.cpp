#include "NoorRay/Materials/Nodes/RampSampler.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "NoorRay/Materials/Nodes/NodeAccess.h"

namespace NoorRay::Materials {

namespace {

constexpr float cardinalTension = 0.71f;

Core::float4 weighted(const std::array<Core::float4, 4>& colors, const std::array<float, 4>& weights) {
    Core::float4 result{};
    for (std::size_t index = 0; index < colors.size(); ++index) {
        result.x += colors[index].x * weights[index];
        result.y += colors[index].y * weights[index];
        result.z += colors[index].z * weights[index];
        result.w += colors[index].w * weights[index];
    }
    return {std::clamp(result.x, 0.0f, 1.0f), std::clamp(result.y, 0.0f, 1.0f), std::clamp(result.z, 0.0f, 1.0f), std::clamp(result.w, 0.0f, 1.0f)};
}

std::array<float, 4> splineWeights(const float t, const bool cardinal) {
    const float t2 = t * t;
    const float t3 = t2 * t;
    if (cardinal) {
        const float c = cardinalTension;
        return {-c * t3 + 2.0f * c * t2 - c * t, (2.0f - c) * t3 + (c - 3.0f) * t2 + 1.0f, (c - 2.0f) * t3 + (3.0f - 2.0f * c) * t2 + c * t, c * t3 - c * t2};
    }
    const float s = 1.0f - t;
    return {s * s * s / 6.0f, 0.5f * t3 - t2 + 2.0f / 3.0f, -0.5f * t3 + 0.5f * t2 + 0.5f * t + 1.0f / 6.0f, t3 / 6.0f};
}

//! The hue of the colour in turns, its largest and its smallest component.
std::array<float, 3> hueRange(const Core::float4 color) {
    const float high = std::max({color.x, color.y, color.z});
    const float low = std::min({color.x, color.y, color.z});
    const float chroma = high - low;
    if (chroma <= 0.0f)
        return {0.0f, high, low};
    float red = color.x;
    float green = color.y;
    float blue = color.z;
    float offset = 0.0f;
    if (green < blue) {
        std::swap(green, blue);
        offset = -1.0f;
    }
    if (red < green) {
        std::swap(red, green);
        offset = -2.0f / 6.0f - offset;
    }
    return {std::abs(offset + (green - blue) / (6.0f * chroma)), high, low};
}

Core::float3 pureHue(const float hue) {
    const auto channel = [&](const float offset) { return std::clamp(std::abs(std::fmod(hue * 6.0f + offset, 6.0f) - 3.0f) - 1.0f, 0.0f, 1.0f); };
    return {channel(0.0f), channel(4.0f), channel(2.0f)};
}

Core::float3 toHsv(const Core::float4 color) {
    const auto [hue, high, low] = hueRange(color);
    return {hue, high > 0.0f ? (high - low) / high : 0.0f, high};
}

Core::float4 fromHsv(const Core::float3 hsv, const float alpha) {
    const Core::float3 pure = pureHue(hsv.x);
    const auto channel = [&](const float value) { return ((value - 1.0f) * hsv.y + 1.0f) * hsv.z; };
    return {channel(pure.x), channel(pure.y), channel(pure.z), alpha};
}

Core::float3 toHsl(const Core::float4 color) {
    const auto [hue, high, low] = hueRange(color);
    const float lightness = 0.5f * (high + low);
    const float chroma = high - low;
    const float divisor = lightness > 0.5f ? 2.0f - high - low : high + low;
    return {hue, chroma > 0.0f && divisor != 0.0f ? chroma / divisor : 0.0f, lightness};
}

Core::float4 fromHsl(const Core::float3 hsl, const float alpha) {
    const Core::float3 pure = pureHue(hsl.x);
    const float chroma = (1.0f - std::abs(2.0f * hsl.z - 1.0f)) * hsl.y;
    const auto channel = [&](const float value) { return (value - 0.5f) * chroma + hsl.z; };
    return {channel(pure.x), channel(pure.y), channel(pure.z), alpha};
}

}

RampSampler::Element RampSampler::element(const std::size_t index) const {
    const std::span<const float> values = elements.subspan(index * rampElementSize, rampElementSize);
    return {values[0], {values[1], values[2], values[3], values[4]}};
}

float RampSampler::blendHue(const float a, const float b, const float factor) const {
    const float difference = b - a;
    float end = b;
    if (hueInterpolation == "NEAR" && std::abs(difference) > 0.5f)
        end = difference > 0.0f ? b - 1.0f : b + 1.0f;
    else if (hueInterpolation == "FAR" && std::abs(difference) < 0.5f)
        end = difference > 0.0f ? b - 1.0f : b + 1.0f;
    else if (hueInterpolation == "CCW" && difference > 0.0f)
        end = b - 1.0f;
    else if (hueInterpolation == "CW" && difference < 0.0f)
        end = b + 1.0f;
    const float hue = std::lerp(a, end, factor);
    return hue - std::floor(hue);
}

Core::float4 RampSampler::blend(const Core::float4 a, const Core::float4 b, const float factor) const {
    const float alpha = std::lerp(a.w, b.w, factor);
    if (colorMode == "RGB")
        return {std::lerp(a.x, b.x, factor), std::lerp(a.y, b.y, factor), std::lerp(a.z, b.z, factor), alpha};
    const bool hsv = colorMode == "HSV";
    const Core::float3 from = hsv ? toHsv(a) : toHsl(a);
    const Core::float3 to = hsv ? toHsv(b) : toHsl(b);
    const Core::float3 mixed{blendHue(from.x, to.x, factor), std::lerp(from.y, to.y, factor), std::lerp(from.z, to.z, factor)};
    return hsv ? fromHsv(mixed, alpha) : fromHsl(mixed, alpha);
}

Core::float4 RampSampler::evaluate(const float position) const {
    const std::size_t count = elements.size() / rampElementSize;
    if (count == 0)
        return {};
    if (count == 1)
        return element(0).color;
    const bool hue = colorMode != "RGB";
    const bool spline = !hue && (interpolation == "B_SPLINE" || interpolation == "CARDINAL");
    std::size_t next = 0;
    while (next < count && element(next).position <= position)
        ++next;
    const Element first = element(0);
    const Element last = element(count - 1);
    if (!spline && position <= first.position)
        return first.color;
    if (!spline && next == count)
        return last.color;
    const Element left = next == 0 ? Element{0.0f, first.color} : element(next - 1);
    const Element right = next == count ? Element{1.0f, last.color} : element(next);
    float factor = next == count ? 0.0f : 1.0f;
    if (left.position != right.position)
        factor = (position - left.position) / (right.position - left.position);
    if (spline) {
        const Core::float4 before = next < 2 ? left.color : element(next - 2).color;
        const Core::float4 after = next + 1 >= count ? right.color : element(next + 1).color;
        return weighted({before, left.color, right.color, after}, splineWeights(std::clamp(factor, 0.0f, 1.0f), interpolation == "CARDINAL"));
    }
    if (!hue && interpolation == "CONSTANT")
        return left.color;
    if (!hue && interpolation == "EASE")
        factor = factor * factor * (3.0f - 2.0f * factor);
    return blend(left.color, right.color, factor);
}

std::vector<float> RampSampler::sample() const {
    const RampPieces pieces = rampPieces(elements);
    std::vector<float> samples;
    samples.reserve((pieces.pieces.size() + 1) * 4);
    const auto add = [&](const float position) {
        const Core::float4 color = evaluate(position);
        samples.insert(samples.end(), {color.x, color.y, color.z, color.w});
    };
    add(pieces.first);
    for (const RampPiece& piece : pieces.pieces)
        add(piece.start + piece.step);
    return samples;
}

}
