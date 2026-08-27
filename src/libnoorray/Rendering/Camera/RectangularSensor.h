#pragma once

#include <glm/common.hpp>

#include "Rendering/Camera/Sensor.h"

class RectangularSensor : public Sensor {
public:
    RectangularSensor() = default;
    explicit RectangularSensor(const Sensor& other);
    ~RectangularSensor() override = default;
    bool renderUi(Sensor& owner);

    // Blends the sample into the running accumulation. The finalize kernel,
    // which owns the frame output surface, writes resolved pixels afterward.
    template <typename WritePixel>
    NR_CPU_GPU void addRgbSample(uint32_t pixel, const glm::vec3 radiance,
        const SensorSampleContext& ctx, const WritePixel& writePixel) const
    {
        const float weight = ctx.totalAccumulated == 0
            ? 1.0f : 1.0f / static_cast<float>(ctx.totalAccumulated + 1);
        const glm::vec4 previous = ctx.accumulation[pixel];
        const glm::vec3 blendedRadiance = glm::mix(
            glm::vec3(previous.x, previous.y, previous.z), radiance, weight);
        const float accAlpha = previous.w * (1.0f - weight) + ctx.alpha * weight;
        const glm::vec4 blended(blendedRadiance, accAlpha);
        ctx.accumulation[pixel] = blended;
        writePixel(blended);
    }

    template <typename WritePixel>
    NR_CPU_GPU void addSample(uint32_t pixel, const SampledSpectrum& L,
        const SampledWavelengths& wl, float, const SensorSampleContext& ctx,
        const WritePixel& writePixel) const
    {
        addRgbSample(pixel,
            sensorRGBFromSpectrum(L, wl, ctx.cieX, ctx.cieY, ctx.cieZ),
            ctx, writePixel);
    }
};
