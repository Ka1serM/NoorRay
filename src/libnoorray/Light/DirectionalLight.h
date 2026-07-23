#pragma once

#include <glm/vec3.hpp>
#include "Light/LightSample.h"
#include "Shading/RgbToSpectrum.h"

class DirectionalLight {
public:
    glm::vec3 direction{0.f, -1.f, 0.f};
    glm::vec3 color{1.f};
    float intensity{1.f};
    float softAngle{0.53f};

    NR_CPU_GPU float selectionWeight() const
    {
        return lightSelectionLuminance(color) * fmaxf(intensity, 0.0f);
    }

    NR_CPU_GPU LightSample sampleLi(
        const glm::vec3 surfacePosition, RandomState& rng,
        const SampledWavelengths& wl,
        const float* spectrumScale, const float* spectrumCoeffs, const float* d65) const
    {
        (void)surfacePosition;
        LightSample sample{};
        const glm::vec3 centerDirection = -glm::normalize(direction);
        sample.direction = centerDirection;
        sample.distance = 1e16f;
        sample.radiance = rgbIlluminantToSpectrum(
            color * intensity, wl, spectrumScale, spectrumCoeffs, d65);
        if (softAngle <= 0.0f)
            return sample;

        // Match Cycles' sun-light convention: the UI value is the angular
        // diameter, while sampling uses its half-angle.
        const float halfAngle = 0.5f * fminf(softAngle, 180.0f)
            * LightPi / 180.0f;
        const UniformConeSample cone = sampleUniformCone(
            centerDirection, oneMinusCosine(halfAngle),
            glm::vec2(randomFloat(rng), randomFloat(rng)));
        sample.direction = cone.direction;
        sample.pdf = cone.pdf;

        // Cycles normalizes the sun disk by its projected area. Store Le/pdf,
        // as the other NoorRay analytic-light samples do.
        const float projectedArea = LightPi * sinf(halfAngle) * sinf(halfAngle);
        const float estimatorScale = 1.0f
            / fmaxf(projectedArea * sample.pdf, 1.0e-20f);
        sample.radiance *= estimatorScale;
        return sample;
    }

    NR_CPU_GPU LightHit intersect(
        const Ray& ray, const SampledWavelengths& wl,
        const float* spectrumScale, const float* spectrumCoeffs,
        const float* d65) const
    {
        LightHit hit{};
        const glm::vec3 centerDirection = -glm::normalize(direction);
        const float halfAngle = 0.5f * fminf(fmaxf(softAngle, 0.0f), 180.0f)
            * LightPi / 180.0f;
        if (halfAngle <= 0.0f) {
            if (glm::dot(ray.direction(), centerDirection) < 1.0f - 1.0e-7f)
                return hit;
            hit.radiance = rgbIlluminantToSpectrum(
                color * intensity, wl, spectrumScale, spectrumCoeffs, d65);
        } else {
            const float oneMinusCosineHalfAngle = oneMinusCosine(halfAngle);
            if (1.0f - glm::dot(ray.direction(), centerDirection)
                > oneMinusCosineHalfAngle)
                return hit;
            hit.pdf = 1.0f / (2.0f * LightPi * oneMinusCosineHalfAngle);
            const float projectedArea =
                LightPi * sinf(halfAngle) * sinf(halfAngle);
            hit.radiance = rgbIlluminantToSpectrum(
                color * (intensity / fmaxf(projectedArea, 1.0e-20f)),
                wl, spectrumScale, spectrumCoeffs, d65);
        }
        hit.distance = Ray::InfiniteDistance;
        return hit;
    }

};
