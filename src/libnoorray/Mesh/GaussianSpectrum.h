#pragma once

#include <glm/vec3.hpp>

#include "Raytracing/RgbToSpectrum.h"

extern const float sRGBToSpectrumTable_Scale[64];
extern const float sRGBToSpectrumTable_Data[3][64][64][64][3];

inline glm::vec3 gaussianSpectrumCoefficients(const glm::vec3 rgb)
{
    glm::vec3 result;
    rgbLookupCoeffs(rgb, sRGBToSpectrumTable_Scale,
        reinterpret_cast<const float*>(sRGBToSpectrumTable_Data), result.x, result.y, result.z);
    return result;
}

inline glm::vec3 gaussianSpectrumDeltaCoefficients(
    const glm::vec3 rgbCoefficient, const glm::vec3 neutral)
{
    return gaussianSpectrumCoefficients(glm::clamp(glm::vec3(0.5f) + rgbCoefficient,
        glm::vec3(0.0f), glm::vec3(1.0f))) - neutral;
}

inline glm::vec3 gaussianSpectrumDeltaCoefficients(const glm::vec3 rgbCoefficient)
{
    return gaussianSpectrumDeltaCoefficients(rgbCoefficient,
        gaussianSpectrumCoefficients(glm::vec3(0.5f)));
}

inline glm::vec3 gaussianSpectrumCoefficientsToRgb(const glm::vec3 coefficients)
{
    glm::vec3 xyz(0.0f);
    for (int sample = 0; sample < NrCIESamples; ++sample)
    {
        const float wavelength = NrLambdaMin + static_cast<float>(sample);
        const float value = rgbSigmoidEval(coefficients.x, coefficients.y, coefficients.z, wavelength);
        xyz += value * glm::vec3(NrCIE_X[sample], NrCIE_Y[sample], NrCIE_Z[sample]);
    }
    xyz /= NrCIE_Y_integral;
    return glm::max(xyzToLinearSRGB(xyz), glm::vec3(0.0f));
}
