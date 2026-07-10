#pragma once

#include <glm/common.hpp>

#include "Camera/Sensor.h"

class RectangularSensor : public Sensor {
public:
#ifndef NR_GPU_CODE
    bool renderUi(Sensor& owner);
#endif

    NR_CPU_GPU void addSample(uint32_t pixel, const SampledSpectrum& L,
        const SampledWavelengths& wl, float, const SensorSampleContext& ctx) const
    {
        glm::vec3 radiance = sensorRGBFromSpectrum(L, wl, ctx.cieX, ctx.cieY, ctx.cieZ);
        const float weight = ctx.totalAccumulated == 0
            ? 1.0f : 1.0f / static_cast<float>(ctx.totalAccumulated + 1);
        const glm::vec4 previous = ctx.accumulation[pixel];
        radiance = glm::mix(glm::vec3(previous.x, previous.y, previous.z), radiance, weight);
        const float accAlpha = previous.w * (1.0f - weight) + ctx.alpha * weight;
        ctx.accumulation[pixel] = glm::vec4(radiance, accAlpha);
    }
};
