#pragma once

#include <glm/vec3.hpp>
#include "Light/LightSample.h"
#include "Raytracing/RgbToSpectrum.h"

class SpotLight {
public:
    glm::vec3 position{};
    glm::vec3 direction{};
    glm::vec3 color{1.f};
    float intensity{1.f};
    float innerConeAngle{20.f};
    float outerConeAngle{30.f};

    NR_CPU_GPU float selectionWeight() const
    {
        const float cone = 2.0f * LightPi * (1.0f - cosf(
            fmaxf(innerConeAngle, outerConeAngle) * LightPi / 180.0f));
        return lightSelectionLuminance(color) * fmaxf(intensity, 0.0f) * cone;
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

        const float coneCos = glm::dot(glm::normalize(direction), -sample.direction);
        const float innerAngle = fminf(innerConeAngle, outerConeAngle);
        const float outerAngle = fmaxf(innerConeAngle, outerConeAngle);
        const float innerCos = cosf(innerAngle * LightPi / 180.0f);
        const float outerCos = cosf(outerAngle * LightPi / 180.0f);
        const float cone = fminf(fmaxf(
            (coneCos - outerCos) / fmaxf(innerCos - outerCos, 1e-5f), 0.0f), 1.0f);
        const float smoothCone = cone * cone * (3.0f - 2.0f * cone);
        const glm::vec3 rgb = color * (intensity * smoothCone /
            fmaxf(sample.distance * sample.distance, 1e-6f));
        sample.radiance = rgbIlluminantToSpectrum(rgb, wl, spectrumScale, spectrumCoeffs, d65);
        return sample;
    }

};
