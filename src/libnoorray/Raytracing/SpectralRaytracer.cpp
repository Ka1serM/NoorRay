#include "SpectralRaytracer.h"

#include <span>

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
    const noorrhi::Shader raygen = load(device, raygenSpv);
    const noorrhi::Shader miss = load(device, missSpv);
    const noorrhi::Shader hit = load(device, hitSpv);
    const noorrhi::Shader emissionHit = load(device, emissionHitSpv);
    const noorrhi::Shader opacityAnyHit = load(device, opacityAnyHitSpv);
    const noorrhi::Shader gaussianAnyHit = load(device, gaussianAnyHitSpv);
    const noorrhi::Shader gaussianHit = load(device, gaussianHitSpv);

    // Two ray types per geometry: primary/shadow and emission lookup. Gaussian
    // instances use SBT offset 2, matching the shared TLAS construction.
    pipeline = device.ray_tracing({raygen, {miss},
        {hit, emissionHit, gaussianHit, gaussianHit},
        {opacityAnyHit, opacityAnyHit, gaussianAnyHit, gaussianAnyHit}, {}});
}
