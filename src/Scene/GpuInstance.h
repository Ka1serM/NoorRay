#pragma once

#include <cstdint>

#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>

struct GpuInstance
{
    glm::mat4 objectToWorld{1.0f};
    glm::mat3 normalToWorld{1.0f};
    uint32_t meshIndex{};
};
