#pragma once

#include <glm/vec3.hpp>
#include "Light/LightSample.h"
#include "Shading/RgbToSpectrum.h"

class SpotLight {
public:
    glm::vec3 position{};
    glm::vec3 direction{};
    glm::vec3 color{1.f};
    float intensity{1.f};
    float softRadius{};
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
        LightSample sample = sampleSphereLight(
            surfacePosition, position, softRadius, rng);
        if (sample.distance <= 0.0f)
            return sample;
        const float cone = coneAttenuation(-sample.direction);
        const glm::vec3 rgb = softRadius > 0.0f
            ? color * (intensity * cone / (LightPi * softRadius * softRadius)
                / fmaxf(sample.pdf, 1.0e-20f))
            : color * (intensity * cone
                / fmaxf(sample.distance * sample.distance, 1e-6f));
        sample.radiance = rgbIlluminantToSpectrum(rgb, wl, spectrumScale, spectrumCoeffs, d65);
        return sample;
    }

    NR_CPU_GPU LightHit intersect(
        const Ray& ray, const float tMin, const float tMax,
        const SampledWavelengths& wl,
        const float* spectrumScale, const float* spectrumCoeffs,
        const float* d65) const
    {
        LightHit hit{};
        if (!intersectSphereLight(ray, tMin, tMax, position, softRadius, hit.distance))
            return {};
        hit.pdf = sphereLightPdf(ray.origin(), position, softRadius);
        const glm::vec3 hitPosition = ray.at(hit.distance);
        const glm::vec3 outgoing = glm::normalize(ray.origin() - hitPosition);
        const glm::vec3 rgb = color * (intensity * coneAttenuation(outgoing)
            / (LightPi * softRadius * softRadius));
        hit.radiance = rgbIlluminantToSpectrum(
            rgb, wl, spectrumScale, spectrumCoeffs, d65);
        return hit;
    }

private:
    NR_CPU_GPU float coneAttenuation(const glm::vec3 outgoing) const
    {
        const float coneCos = glm::dot(glm::normalize(direction), outgoing);
        const float innerAngle = fminf(innerConeAngle, outerConeAngle);
        const float outerAngle = fmaxf(innerConeAngle, outerConeAngle);
        const float innerCos = cosf(innerAngle * LightPi / 180.0f);
        const float outerCos = cosf(outerAngle * LightPi / 180.0f);
        const float cone = fminf(fmaxf(
            (coneCos - outerCos) / fmaxf(innerCos - outerCos, 1e-5f), 0.0f), 1.0f);
        return cone * cone * (3.0f - 2.0f * cone);
    }

};
