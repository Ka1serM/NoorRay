#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "NoorRay/Core/Math.h"

namespace NoorRay::Materials {

//! Evaluates a colour ramp as Blender does.
class RampSampler {
public:
    //! elements holds the position and colour of each element, sorted by position.
    RampSampler(std::span<const float> elements, std::string_view interpolation, std::string_view colorMode, std::string_view hueInterpolation)
        : elements(elements), interpolation(interpolation), colorMode(colorMode), hueInterpolation(hueInterpolation) {}

    Core::float4 evaluate(float position) const;
    //! The values of color_ramp.samples: four per piece end of rampPieces.
    std::vector<float> sample() const;

private:
    struct Element {
        float position = 0.0f;
        Core::float4 color;
    };

    Element element(std::size_t index) const;
    float blendHue(float a, float b, float factor) const;
    Core::float4 blend(Core::float4 a, Core::float4 b, float factor) const;

    std::span<const float> elements;
    std::string_view interpolation;
    std::string_view colorMode;
    std::string_view hueInterpolation;
};

}
