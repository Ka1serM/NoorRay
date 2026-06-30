#pragma once

#include <glm/vec3.hpp>
#include "Light/LightSample.h"
#include "Raytracing/RgbToSpectrum.h"

class PointLight {
public:
    glm::vec3 position{};
    glm::vec3 color{1.f};
    float intensity{1.f};

    NR_CPU_GPU float selectionWeight() const
    {
        return lightSelectionLuminance(color) * fmaxf(intensity, 0.0f) * (4.0f * LightPi);
    }

    NR_CPU_GPU LightSample sampleLi(
        const glm::vec3 surfacePosition, RandomState& rng,
        const SampledWavelengths& wl,
        const float* spectrumScale, const float* spectrumCoeffs, const float* d65) const
    {
        (void)rng;
        LightSample sample{};
        const glm::vec3 delta = position - surfacePosition;
        sample.distance = glm::length(delta);
        if (sample.distance <= 0.0f)
            return sample;
        sample.direction = delta / sample.distance;
        const glm::vec3 rgb = color * (intensity /
            fmaxf(sample.distance * sample.distance, 1e-6f));
        sample.radiance = rgbIlluminantToSpectrum(rgb, wl, spectrumScale, spectrumCoeffs, d65);
        return sample;
    }

    bool renderUi();
};
