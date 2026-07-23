#pragma once

#include <glm/vec3.hpp>
#include "Light/LightSample.h"
#include "Shading/RgbToSpectrum.h"

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
        LightSample sample = sampleSphereLight(
            surfacePosition, position, softRadius, rng);
        if (sample.distance <= 0.0f)
            return sample;
        const glm::vec3 rgb = softRadius > 0.0f
            ? color * (intensity / (LightPi * softRadius * softRadius)
                / fmaxf(sample.pdf, 1.0e-20f))
            : color * (intensity
                / fmaxf(sample.distance * sample.distance, 1e-6f));
        sample.radiance = rgbIlluminantToSpectrum(rgb, wl, spectrumScale, spectrumCoeffs, d65);
        return sample;
    }

    NR_CPU_GPU LightHit intersect(
        const Ray& ray, const SampledWavelengths& wl,
        const float* spectrumScale, const float* spectrumCoeffs,
        const float* d65) const
    {
        LightHit hit{};
        if (!intersectSphereLight(ray, position, softRadius, hit.distance))
            return {};
        hit.pdf = sphereLightPdf(ray.origin(), position, softRadius);
        const glm::vec3 rgb = color
            * (intensity / (LightPi * softRadius * softRadius));
        hit.radiance = rgbIlluminantToSpectrum(
            rgb, wl, spectrumScale, spectrumCoeffs, d65);
        return hit;
    }

};
