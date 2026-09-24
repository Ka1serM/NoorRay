#include "SpectralRaytracer.h"

#include <cstdint>
#include <span>
#include <vector>

#include "Materials/Shading/ShadingTables.h"

namespace
{
// The SPIR-V is compiled into the build tree; --embed-dir points #embed there.
alignas(uint32_t) constexpr unsigned char raygenSpv[] = {
    #embed "Raytracer/Raytracer.spv"
};
alignas(uint32_t) constexpr unsigned char missSpv[] = {
    #embed "Raytracer/RaytracingMiss.spv"
};
alignas(uint32_t) constexpr unsigned char hitSpv[] = {
    #embed "Raytracer/RaytracingHit.spv"
};
alignas(uint32_t) constexpr unsigned char emissionHitSpv[] = {
    #embed "Raytracer/EmissionHit.spv"
};
alignas(uint32_t) constexpr unsigned char opacityAnyHitSpv[] = {
    #embed "Raytracer/OpacityAnyHit.spv"
};
alignas(uint32_t) constexpr unsigned char gaussianAnyHitSpv[] = {
    #embed "Raytracer/GaussianAnyHit.spv"
};
alignas(uint32_t) constexpr unsigned char gaussianHitSpv[] = {
    #embed "Raytracer/GaussianHit.spv"
};

template<class ShaderArray>
noorrhi::Shader load(noorrhi::Device& device, const ShaderArray& bytes)
{
    return device.create_shader(std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(bytes), sizeof(bytes)), "main");
}
}

SpectralRaytracer::SpectralRaytracer(noorrhi::Device& device,
    const uint32_t width, const uint32_t height, const bool exportColorMemory)
    : Raytracer(device, width, height, exportColorMemory)
{
    raygen = load(device, raygenSpv);
    const noorrhi::Shader miss = load(device, missSpv);
    const noorrhi::Shader hit = load(device, hitSpv);
    const noorrhi::Shader emissionHit = load(device, emissionHitSpv);
    const noorrhi::Shader opacityAnyHit = load(device, opacityAnyHitSpv);
    const noorrhi::Shader gaussianAnyHit = load(device, gaussianAnyHitSpv);
    const noorrhi::Shader gaussianHit = load(device, gaussianHitSpv);

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
