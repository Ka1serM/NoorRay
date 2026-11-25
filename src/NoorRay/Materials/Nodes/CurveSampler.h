#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "NoorRay/Core/Math.h"

namespace NoorRay::Materials {

enum class CurveHandle : std::uint8_t { Auto, AutoClamped, Vector };

//! A curve of a curve mapping, evaluated through the table of 256 segments that Blender makes from its Bezier segments.
class CurveSampler {
public:
    //! points holds x and y, sorted by x; low and high are the x clip range of the mapping.
    CurveSampler(std::span<const float> points, std::span<const CurveHandle> handles, bool extrapolates, float low, float high);

    float evaluate(float x) const;

private:
    struct Knot {
        Core::float2 left;
        Core::float2 center;
        Core::float2 right;
    };

    static constexpr std::size_t tableSize = 256;

    static std::vector<Knot> makeKnots(std::span<const float> points, std::span<const CurveHandle> handles);
    static std::vector<Core::float2> makePolyline(std::span<const Knot> knots);
    float extend(float x) const;
    float extend(float x, bool below, Core::float2 end) const;

    bool extrapolates;
    float tableLow;
    float tableHigh;
    Core::float2 first;
    Core::float2 last;
    Core::float2 slopeIn{0.0f, 0.0f};
    Core::float2 slopeOut{0.0f, 0.0f};
    std::array<float, tableSize + 1> table{};
};

}
