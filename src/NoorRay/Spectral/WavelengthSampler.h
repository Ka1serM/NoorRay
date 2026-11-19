#pragma once

#include <cmath>

namespace NoorRay::Spectral {

inline constexpr float visibleLambdaMin = 360.0f;
inline constexpr float visibleLambdaMax = 830.0f;
inline constexpr unsigned wavelengthCount = 4;

//! The analytic visible-range distribution of pbrt-v4 (Radziszewski et al. 2009).
inline float sampleVisibleWavelength(const float u) {
    return 538.0f - 138.888889f * std::atanh(0.85691062f - 1.82750197f * u);
}

inline float visibleWavelengthPdf(const float lambda) {
    if (lambda < visibleLambdaMin || lambda > visibleLambdaMax)
        return 0.0f;
    const float c = std::cosh(0.0072f * (lambda - 538.0f));
    return 0.0039398042f / (c * c);
}

inline float wavelengthOffset(const float u, const unsigned index) {
    const float shifted = u + static_cast<float>(index) / static_cast<float>(wavelengthCount);
    return shifted - std::floor(shifted);
}

}
