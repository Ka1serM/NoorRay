#pragma once

#include <cmath>

#include "NoorRay/Core/Math.h"

namespace NoorRay::Spectral {

//! Planck's law in W / (m^2 sr m) for a wavelength in nanometres.
inline float planck(const float lambdaNm, const float kelvin) {
    if (kelvin <= 0.0f)
        return 0.0f;
    const double c = 299792458.0, h = 6.62606957e-34, kb = 1.3806488e-23;
    const double l = lambdaNm * 1e-9;
    return static_cast<float>((2.0 * h * c * c) / (l * l * l * l * l * (std::exp((h * c) / (l * kb * kelvin)) - 1.0)));
}

//! The linear sRGB colour of a blackbody, scaled so that its luminance is one.
Core::float3 blackbodyColor(float kelvin);

}
