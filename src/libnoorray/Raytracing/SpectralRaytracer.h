#pragma once

#include "Raytracer.h"

// Reference spectral path tracer. Shared scene, resource, AOV and readback
// behavior lives in Raytracer; this class only supplies the spectral pipeline.
class SpectralRaytracer final : public Raytracer
{
public:
    SpectralRaytracer(noorrhi::Device& device, uint32_t width, uint32_t height,
        bool exportColorMemory = false);

    RaytracerType type() const noexcept override
    {
        return RaytracerType::Spectral;
    }
};
