#pragma once

#include <glm/vec3.hpp>
#include "Light/LightSample.h"

class PointLight {
public:
    glm::vec3 position{};
    glm::vec3 color{1.f};
    float intensity{1.f};

    NR_CPU_GPU LightSample sampleLi(const glm::vec3 surfacePosition, uint32_t& rng) const
    {
        (void)rng;
        LightSample sample{};
        const glm::vec3 delta = position - surfacePosition;
        sample.distance = glm::length(delta);
        if (sample.distance <= 0.0f)
            return sample;
        sample.direction = delta / sample.distance;
        sample.radiance = color * (intensity /
            fmaxf(sample.distance * sample.distance, 1e-6f));
        return sample;
    }

    bool renderUi();
};
