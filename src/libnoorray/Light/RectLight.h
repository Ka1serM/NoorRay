#pragma once

#include <glm/vec3.hpp>
#include "Light/LightSample.h"
#include "Shading/RgbToSpectrum.h"

class RectLight {
public:
    glm::vec3 position{};
    glm::vec3 direction{};
    glm::vec3 tangent{1.f, 0.f, 0.f};
    glm::vec3 color{1.f};
    float intensity{1.f};
    float width{1.f};
    float height{1.f};
    int twoSided{};
    float barnDoorAngle{90.f};
    float barnDoorLength{};

    NR_CPU_GPU float selectionWeight() const
    {
        const float sidedness = twoSided != 0 ? 2.0f : 1.0f;
        return lightSelectionLuminance(color) * fmaxf(intensity, 0.0f)
            * fmaxf(width * height, 0.0f) * LightPi * sidedness;
    }

    NR_CPU_GPU LightSample sampleLi(
        const glm::vec3 surfacePosition, RandomState& rng,
        const SampledWavelengths& wl,
        const float* spectrumScale, const float* spectrumCoeffs, const float* d65) const
    {
        LightSample sample{};
        const glm::vec3 normal = glm::normalize(direction);
        const glm::vec3 u = glm::normalize(tangent);
        const glm::vec3 v = glm::normalize(glm::cross(normal, u));
        glm::vec3 sampledPosition = position;
        sample.pdf = sampleSphericalRectangle(surfacePosition,
            sampledPosition, u, width, v, height,
            glm::vec2(randomFloat(rng), randomFloat(rng)));
        if (sample.pdf <= 0.0f)
            return sample;
        const glm::vec3 delta = sampledPosition - surfacePosition;
        sample.distance = glm::length(delta);
        if (sample.distance <= 0.0f)
            return sample;
        sample.direction = delta / sample.distance;

        float emitterCosine = glm::dot(normal, -sample.direction);
        emitterCosine = twoSided != 0 ? fabsf(emitterCosine) : fmaxf(emitterCosine, 0.0f);
        if (emitterCosine <= 0.0f)
            return sample;
        float barnDoorMask = 1.0f;
        if (barnDoorLength > 0.0f && barnDoorAngle < 89.9f && emitterCosine > 0.0f)
        {
            const glm::vec3 outgoing = -sample.direction;
            const float forward = fmaxf(fabsf(glm::dot(normal, outgoing)), 1e-5f);
            const float localU = glm::dot(sampledPosition - position, u);
            const float localV = glm::dot(sampledPosition - position, v);
            const float projectedU = localU
                + barnDoorLength * glm::dot(outgoing, u) / forward;
            const float projectedV = localV
                + barnDoorLength * glm::dot(outgoing, v) / forward;
            const float expansion = barnDoorLength * tanf(
                fmaxf(barnDoorAngle, 0.0f) * LightPi / 180.0f);
            if (fabsf(projectedU) > width * 0.5f + expansion
                || fabsf(projectedV) > height * 0.5f + expansion)
                barnDoorMask = 0.0f;
        }
        const glm::vec3 rgb = color * (intensity * barnDoorMask
            / fmaxf(sample.pdf, 1.0e-20f));
        sample.radiance = rgbIlluminantToSpectrum(rgb, wl, spectrumScale, spectrumCoeffs, d65);
        return sample;
    }

};
