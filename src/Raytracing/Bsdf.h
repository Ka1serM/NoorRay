#pragma once

#include <cstdint>

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
    SampledSpectrum emission{};
    float pdf{};
    float eta{1.0f};
    BsdfEvent event{BsdfEvent::Diffuse};
};
