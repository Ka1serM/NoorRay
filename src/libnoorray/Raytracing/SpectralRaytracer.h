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

protected:
    void renderImpl() override;
    void onHitRecordsChanged(std::span<const HitRecord> records) override;

private:
    noorrhi::Shader raygen;
    noorrhi::RayTracingLibrary library;
    noorrhi::RayTracingPipeline pipeline;
    // Read only by the spectral closures: energy-compensation LUTs and the
    // CIE/D65 and RGB-to-spectrum tables.
    noorrhi::Buffer<std::uint16_t> energyLuts;
    noorrhi::Buffer<float> spectralTables;
};
