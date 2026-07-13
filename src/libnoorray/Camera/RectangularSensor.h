#pragma once

#include <glm/common.hpp>

#include "Camera/Sensor.h"

class RectangularSensor : public Sensor {
public:
#ifndef NR_GPU_CODE
    RectangularSensor() = default;
    explicit RectangularSensor(const Sensor& other);
    ~RectangularSensor() override = default;
    bool renderUi(Sensor& owner);
#endif

    // Blends the sample into the running accumulation. The finalize kernel,
    // which owns the frame output surface, writes resolved pixels afterward.
    NR_CPU_GPU void addSample(uint32_t pixel, const SampledSpectrum& L,
        const SampledWavelengths& wl, float, const SensorSampleContext& ctx) const
    {
        glm::vec3 radiance = sensorRGBFromSpectrum(L, wl, ctx.cieX, ctx.cieY, ctx.cieZ);
        updateNoiseMoments(pixel, radiance, ctx);
        const float weight = ctx.totalAccumulated == 0
            ? 1.0f : 1.0f / static_cast<float>(ctx.totalAccumulated + 1);
        const glm::vec4 previous = ctx.accumulation[pixel];
        radiance = glm::mix(glm::vec3(previous.x, previous.y, previous.z), radiance, weight);
        const float accAlpha = previous.w * (1.0f - weight) + ctx.alpha * weight;
        const glm::vec4 blended(radiance, accAlpha);
        ctx.accumulation[pixel] = blended;
    }
};
