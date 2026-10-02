#pragma once

#include "Raytracer.h"

// Reference spectral path tracer. It owns its pipeline and composes the
// scene/output resources it needs; it does not inherit resource setup.
class SpectralRaytracer final : public Raytracer
{
public:
    SpectralRaytracer(noorrhi::Device& device, uint32_t width, uint32_t height,
        bool exportColorMemory = false);
    RaytracerResources& resources() override { return common; }
    const RaytracerResources& resources() const override { return common; }
    bool prepareFrameResources() override { return common.prepareFrameResources(); }
    void render(uint32_t frameIndex = 0, uint32_t sampleIndex = 0) override;

    RaytracerType type() const noexcept override
    {
        return RaytracerType::Spectral;
    }

protected:
    void renderImpl();
    void onHitRecordsChanged(std::span<const HitRecord> records);

private:
    RaytracerResources common;
    noorrhi::Shader raygen;
    noorrhi::RayTracingLibrary library;
    noorrhi::RayTracingPipeline pipeline;
    // Read only by the spectral closures: energy-compensation LUTs and the
    // CIE/D65 and RGB-to-spectrum tables.
    noorrhi::Buffer<std::uint16_t> energyLuts;
    noorrhi::Buffer<float> spectralTables;
};
