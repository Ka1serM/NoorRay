#include "NoorRay/Materials/Nodes/CurveSampler.h"

#include <algorithm>
#include <cmath>

namespace NoorRay::Materials {

namespace {

constexpr std::size_t segmentSteps = 32;
//! The scale that makes the automatic handles of four points on a circle follow the circle.
constexpr float autoHandleScale = 2.5614f;

float length(const Core::float2 v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

Core::float2 normalized(const Core::float2 v) {
    const float size = length(v);
    return size == 0.0f ? v : Core::float2{v.x / size, v.y / size};
}

Core::float2 add(const Core::float2 a, const Core::float2 b, const float scale) {
    return {a.x + b.x * scale, a.y + b.y * scale};
}

}

std::vector<CurveSampler::Knot> CurveSampler::makeKnots(const std::span<const float> points, const std::span<const CurveHandle> handles) {
    const std::size_t count = points.size() / 2;
    const auto point = [&](const std::size_t index) { return Core::float2{points[index * 2], points[index * 2 + 1]}; };
    std::vector<Knot> knots(count);
    for (std::size_t index = 0; index < count; ++index) {
        const Core::float2 center = point(index);
        const Core::float2 previous = index > 0 ? point(index - 1) : add(add(center, center, 1.0f), point(index + 1), -1.0f);
        const Core::float2 next = index + 1 < count ? point(index + 1) : add(add(center, center, 1.0f), previous, -1.0f);
        const Core::float2 before = add(center, previous, -1.0f);
        const Core::float2 after = add(next, center, -1.0f);
        Knot& knot = knots[index];
        knot.center = center;
        if (handles[index] == CurveHandle::Vector) {
            knot.left = add(center, before, -1.0f / 3.0f);
            knot.right = add(center, after, 1.0f / 3.0f);
            continue;
        }
        const float beforeLength = length(before) == 0.0f ? 1.0f : length(before);
        const float afterLength = length(after) == 0.0f ? 1.0f : length(after);
        const Core::float2 tangent{after.x / afterLength + before.x / beforeLength, after.y / afterLength + before.y / beforeLength};
        const float scale = length(tangent) * autoHandleScale;
        knot.left = center;
        knot.right = center;
        if (scale == 0.0f)
            continue;
        knot.left = add(center, tangent, -beforeLength / scale);
        knot.right = add(center, tangent, afterLength / scale);
        if (handles[index] != CurveHandle::AutoClamped || index == 0 || index + 1 == count)
            continue;
        const float risePrevious = previous.y - center.y;
        const float riseNext = next.y - center.y;
        if ((risePrevious <= 0.0f && riseNext <= 0.0f) || (risePrevious >= 0.0f && riseNext >= 0.0f)) {
            knot.left.y = center.y;
            knot.right.y = center.y;
            continue;
        }
        const bool falls = risePrevious > 0.0f;
        knot.left.y = falls ? std::min(knot.left.y, previous.y) : std::max(knot.left.y, previous.y);
        knot.right.y = falls ? std::max(knot.right.y, next.y) : std::min(knot.right.y, next.y);
    }
    return knots;
}

std::vector<Core::float2> CurveSampler::makePolyline(const std::span<const Knot> knots) {
    std::vector<Core::float2> polyline;
    polyline.reserve(knots.size() * segmentSteps + 1);
    for (std::size_t index = 0; index + 1 < knots.size(); ++index) {
        const Core::float2 start = knots[index].center;
        const Core::float2 end = knots[index + 1].center;
        Core::float2 startHandle = add(start, knots[index].right, -1.0f);
        Core::float2 endHandle = add(end, knots[index + 1].left, -1.0f);
        const float span = end.x - start.x;
        const float reach = std::abs(startHandle.x) + std::abs(endHandle.x);
        const float scale = reach > span && reach != 0.0f ? span / reach : 1.0f;
        const Core::float2 a = add(start, startHandle, -scale);
        const Core::float2 b = add(end, endHandle, -scale);
        for (std::size_t step = 0; step < segmentSteps; ++step) {
            const float t = static_cast<float>(step) / static_cast<float>(segmentSteps - 1);
            const float s = 1.0f - t;
            const std::array<float, 4> w{s * s * s, 3.0f * s * s * t, 3.0f * s * t * t, t * t * t};
            polyline.push_back({w[0] * start.x + w[1] * a.x + w[2] * b.x + w[3] * end.x, w[0] * start.y + w[1] * a.y + w[2] * b.y + w[3] * end.y});
        }
    }
    return polyline;
}

CurveSampler::CurveSampler(const std::span<const float> points, const std::span<const CurveHandle> handles, const bool extrapolates, const float low,
                           const float high)
    : extrapolates(extrapolates), tableLow(low), tableHigh(high) {
    const std::size_t count = points.size() / 2;
    if (count == 0)
        return;
    first = {points[0], points[1]};
    last = {points[(count - 1) * 2], points[(count - 1) * 2 + 1]};
    if (count == 1) {
        table.fill(first.y);
        return;
    }
    std::vector<Knot> knots = makeKnots(points, handles);
    //! Blender aims an automatic end handle at the near handle of the neighbour, and keeps its length.
    const auto aim = [](Knot& knot, Core::float2 target, const bool right) {
        Core::float2& handle = right ? knot.right : knot.left;
        Core::float2& opposite = right ? knot.left : knot.right;
        const float reach = length(add(knot.center, handle, -1.0f));
        target.x = right ? std::max(target.x, knot.center.x) : std::min(target.x, knot.center.x);
        const Core::float2 direction = add(target, knot.center, -1.0f);
        const float distance = length(direction);
        if (distance <= 1.1920929e-07f)
            return;
        handle = add(knot.center, direction, reach / distance);
        opposite = add(knot.center, direction, -reach / distance);
    };
    if (count > 2) {
        if (handles[0] == CurveHandle::Auto)
            aim(knots.front(), knots[1].left, true);
        if (handles[count - 1] == CurveHandle::Auto)
            aim(knots.back(), knots[count - 2].right, false);
    }
    slopeIn = normalized(add(knots.front().left, knots.front().center, -1.0f));
    slopeOut = normalized(add(knots.back().right, knots.back().center, -1.0f));
    tableLow = std::min(low, first.x);
    tableHigh = std::max(high, last.x);
    const std::vector<Core::float2> polyline = makePolyline(knots);
    std::size_t segment = 0;
    for (std::size_t index = 0; index <= tableSize; ++index) {
        const float x = tableLow + (tableHigh - tableLow) * static_cast<float>(index) / static_cast<float>(tableSize);
        if (x < first.x || x > last.x) {
            table[index] = extend(x);
            continue;
        }
        while (segment + 2 < polyline.size() && x > polyline[segment + 1].x)
            ++segment;
        const Core::float2 a = polyline[segment];
        const Core::float2 b = polyline[segment + 1];
        table[index] = b.x == a.x ? a.y : std::lerp(a.y, b.y, (x - a.x) / (b.x - a.x));
    }
}

float CurveSampler::extend(const float x) const {
    const bool below = x <= first.x;
    return extend(x, below, below ? first : last);
}

float CurveSampler::extend(const float x, const bool below, const Core::float2 end) const {
    const Core::float2 slope = below ? slopeIn : slopeOut;
    if (!extrapolates)
        return end.y;
    if (slope.x == 0.0f)
        return end.y + slope.y * 10000.0f;
    return end.y + slope.y / slope.x * (x - end.x);
}

float CurveSampler::evaluate(const float x) const {
    const float position = (x - tableLow) / (tableHigh - tableLow) * static_cast<float>(tableSize);
    if (position < 0.0f || position > static_cast<float>(tableSize))
        return extend(x, position < 0.0f, position < 0.0f ? Core::float2{tableLow, table.front()} : Core::float2{tableHigh, table.back()});
    const auto index = static_cast<std::size_t>(position);
    if (index >= tableSize)
        return table[tableSize];
    return std::lerp(table[index], table[index + 1], position - static_cast<float>(index));
}

}
