#pragma once

#include <glm/vec3.hpp>
#include "Light/LightSample.h"

class DirectionalLight {
public:
    glm::vec3 direction{0.f, 0.f, 1.f};
    glm::vec3 color{1.f};
    float intensity{1.f};
    float softAngle{0.53f};

    NR_CPU_GPU LightSample sampleLi(const glm::vec3 surfacePosition, uint32_t& rng) const
    {
        (void)surfacePosition;
        LightSample sample{};
        const glm::vec3 centerDirection = -glm::normalize(direction);
        sample.direction = centerDirection;
        sample.distance = 1e16f;
        sample.radiance = color * intensity;
        if (softAngle <= 0.0f)
            return sample;

        glm::vec3 tangent{}, bitangent{};
        makeLightBasis(centerDirection, tangent, bitangent);
        const float maxAngle = fminf(softAngle, 90.0f) * LightPi / 180.0f;
        const float cosTheta = 1.0f - randomFloat(rng) * (1.0f - cosf(maxAngle));
        const float sinTheta = sqrtf(fmaxf(1.0f - cosTheta * cosTheta, 0.0f));
        const float phi = 2.0f * LightPi * randomFloat(rng);
        sample.direction = glm::normalize(
            centerDirection * cosTheta
            + tangent * (sinTheta * cosf(phi))
            + bitangent * (sinTheta * sinf(phi)));
        return sample;
    }

    bool renderUi();
};
