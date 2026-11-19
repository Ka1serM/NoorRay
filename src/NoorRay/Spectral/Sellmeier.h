#pragma once

#include <cmath>

namespace NoorRay::Spectral {

//! Coefficients for wavelengths in micrometres.
struct Sellmeier {
    float b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    float c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    float evaluate(const float lambdaNm) const {
        const float l2 = lambdaNm * lambdaNm * 1e-6f;
        return std::sqrt(1.0f + b1 * l2 / (l2 - c1) + b2 * l2 / (l2 - c2) + b3 * l2 / (l2 - c3));
    }
};

inline constexpr Sellmeier bk7{1.03961212f, 0.231792344f, 1.01046945f, 0.00600069867f, 0.0200179144f, 103.560653f};

//! n = a + b / lambda^2 with lambda in micrometres. Blender's Principled BSDF defines its Dispersion input (1 / Abbe number) this way.
struct Cauchy {
    float a = 1.5f;
    float b = 0.0f;

    static Cauchy fromAbbe(const float ior, const float inverseAbbe) {
        const float b = (ior - 1.0f) * inverseAbbe / (1.0f / (0.4861f * 0.4861f) - 1.0f / (0.6563f * 0.6563f));
        return {ior - b / (0.5876f * 0.5876f), b};
    }
    float evaluate(const float lambdaNm) const {
        const float l = lambdaNm * 1e-3f;
        return a + b / (l * l);
    }
};

}
