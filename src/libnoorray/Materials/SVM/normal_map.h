#pragma once

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "Backend/CUDA/Annotations.h"
#include "Rendering/Ray.h"

namespace nr::svm::detail
{

// MaterialX stdlib/genglsl/mx_normalmap.glsl:
// unpack tangent-space normal, apply the authored tangent frame and scale,
// then normalize. The frame is intentionally not orthogonalized here.
NR_GPU inline glm::vec3 materialXNormalMap(
    glm::vec3 value, const glm::vec2 scale,
    const glm::vec3 normal, const glm::vec3 tangent,
    const glm::vec3 bitangent)
{
    value = glm::dot(value, value) == 0.0f
        ? glm::vec3(0.0f, 0.0f, 1.0f) : value * 2.0f - glm::vec3(1.0f);
    value = tangent * value.x * scale.x
        + bitangent * value.y * scale.y + normal * value.z;
    return nr::safeNormalize(value, normal);
}

} // namespace nr::svm::detail
