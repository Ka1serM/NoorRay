#pragma once

#include <cstdint>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "Raytracing/Spectrum.h"

enum class BsdfEvent : uint32_t
{
    Diffuse,
    Specular,
    Transmission,
};

struct BsdfSample
{
    glm::vec3 direction{};
    SampledSpectrum weight{};
    SampledSpectrum albedo{};
    SampledSpectrum emission{};
    float metallic{};
    float roughness{};
    float specular{};
    float transmission{};
    float pdf{};
    float eta{1.0f};
    bool terminateSecondaryWavelengths{};
    BsdfEvent event{BsdfEvent::Diffuse};
};
