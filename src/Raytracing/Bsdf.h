#pragma once

#include <cstdint>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"

static constexpr float BsdfPi = 3.14159265358979323846f;
static constexpr float BsdfEpsilon = 1e-5f;

enum class BsdfEvent : uint32_t
{
    Diffuse,
    Specular,
    Transmission,
};

struct BsdfSample
{
    glm::vec3 direction{};
    glm::vec3 weight{};
    glm::vec3 albedo{};
    glm::vec3 emission{};
    float metallic{};
    float roughness{};
    float specular{};
    float transmission{};
    BsdfEvent event{BsdfEvent::Diffuse};
};
