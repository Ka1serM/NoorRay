#pragma once

#include <glm/vec3.hpp>
#include "Light/LightSample.h"
#include "Raytracing/RgbToSpectrum.h"

class PointLight {
public:
    glm::vec3 position{};
    glm::vec3 color{1.f};
    float intensity{1.f};
    float softRadius{};

    NR_CPU_GPU float selectionWeight() const
    {
        return lightSelectionLuminance(color) * fmaxf(intensity, 0.0f) * (4.0f * LightPi);
    }

    NR_CPU_GPU LightSample sampleLi(
        const glm::vec3 surfacePosition, RandomState& rng,
        const SampledWavelengths& wl,
        const float* spectrumScale, const float* spectrumCoeffs, const float* d65) const
    {
        LightSample sample{};
        glm::vec3 sampledPosition = position;
        if (softRadius > 0.0f)
        {
            const float z = 1.0f - 2.0f * randomFloat(rng);
            const float r = sqrtf(fmaxf(1.0f - z * z, 0.0f));
            const float phi = 2.0f * LightPi * randomFloat(rng);
            float sinPhi = 0.0f;
            float cosPhi = 1.0f;
            sincosf(phi, &sinPhi, &cosPhi);
            sampledPosition += softRadius * glm::vec3(r * cosPhi, z, r * sinPhi);
        }

        const glm::vec3 delta = sampledPosition - surfacePosition;
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
