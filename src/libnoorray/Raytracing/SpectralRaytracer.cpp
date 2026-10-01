#include "SpectralRaytracer.h"

#include <cstdint>
#include <span>
#include <vector>

#include "Materials/Shading/ShadingTables.h"
#include "Realtime/ShaderLoading.h"

namespace
{
constexpr const char* raygenSpv = "Raytracer/Raytracer.spv";
constexpr const char* missSpv = "Raytracer/RaytracingMiss.spv";
constexpr const char* hitSpv = "Raytracer/RaytracingHit.spv";
constexpr const char* emissionHitSpv = "Raytracer/EmissionHit.spv";
constexpr const char* opacityAnyHitSpv = "Raytracer/OpacityAnyHit.spv";
constexpr const char* gaussianAnyHitSpv = "Raytracer/GaussianAnyHit.spv";
constexpr const char* gaussianHitSpv = "Raytracer/GaussianHit.spv";
}

SpectralRaytracer::SpectralRaytracer(noorrhi::Device& device,
    const uint32_t width, const uint32_t height, const bool exportColorMemory)
    : Raytracer(device, width, height, exportColorMemory, FullOutputAovs::Written)
{
    raygen = loadShader(device, raygenSpv);
    const noorrhi::Shader miss = loadShader(device, missSpv);
    const noorrhi::Shader hit = loadShader(device, hitSpv);
    const noorrhi::Shader emissionHit = loadShader(device, emissionHitSpv);
    const noorrhi::Shader opacityAnyHit = loadShader(device, opacityAnyHitSpv);
    const noorrhi::Shader gaussianAnyHit = loadShader(device, gaussianAnyHitSpv);
    const noorrhi::Shader gaussianHit = loadShader(device, gaussianHitSpv);

    // Two ray types per geometry: primary/shadow and emission lookup. Gaussian
    // instances use SBT offset 2, matching the shared TLAS construction.
    library = device.ray_tracing_library({{raygen}, {miss},
        {hit, emissionHit, gaussianHit, gaussianHit},
        {opacityAnyHit, opacityAnyHit, gaussianAnyHit, gaussianAnyHit}, {}},
        {sizeof(nr::graphics::RayPayload)});
    pipeline = device.ray_tracing(std::span(&library, 1), {});

    const std::vector<std::uint16_t> lut = nr::shading::packEnergyLutTables();
    energyLuts = device.buffer<std::uint16_t>(lut.size());
    energyLuts.upload(std::span<const std::uint16_t>(lut));
    data.energyLuts = energyLuts.ptr().address;
    const std::vector<float> tables = nr::shading::packSpectralTables();
    spectralTables = device.buffer<float>(tables.size());
    spectralTables.upload(std::span<const float>(tables));
    data.spectralTables = spectralTables.ptr().address;
}

// Every section's records use the two mesh groups, whatever the material.
void SpectralRaytracer::onHitRecordsChanged(const std::span<const HitRecord> records)
{
    std::vector<std::uint32_t> groups;
    groups.reserve(records.size());
    for (const HitRecord& record : records)
        groups.push_back(record.rayType + (record.kind == HitRecord::Kind::Gaussian ? 2u : 0u));
    pipeline = renderDevice().ray_tracing(std::span(&library, 1), groups);
}

void SpectralRaytracer::renderImpl()
{
    pipeline.trace(raygen, {logicalRenderWidth(), logicalRenderHeight(), 1}, data);
}
