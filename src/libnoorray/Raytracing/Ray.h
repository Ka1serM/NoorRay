#pragma once

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"

// Geometric ray: only origin and direction. Traversal bounds are passed
// explicitly to intersection functions.
class Ray
{
public:
    static constexpr float DefaultMinDistance = 0.001f;
    static constexpr float DefaultMaxDistance = 1000.0f;
    static constexpr float InfiniteDistance = 1.0e30f;

    NR_CPU_GPU Ray() {}

    NR_CPU_GPU Ray(
        const glm::vec3& origin,
        const glm::vec3& direction)
        : rayOrigin(origin),
          rayDirection(direction)
    {}

    NR_CPU_GPU const glm::vec3& origin() const { return rayOrigin; }
    NR_CPU_GPU const glm::vec3& direction() const { return rayDirection; }

    NR_CPU_GPU glm::vec3 at(const float distance) const
    {
        return rayOrigin + rayDirection * distance;
    }

    NR_CPU_GPU float closestDistanceTo(const glm::vec3& point) const
    {
        const float directionLengthSquared = glm::dot(rayDirection, rayDirection);
        return directionLengthSquared > 0.0f
            ? glm::dot(point - rayOrigin, rayDirection) / directionLengthSquared
            : 0.0f;
    }

    NR_CPU_GPU static Ray fromOffset(
        const glm::vec3& point,
        const glm::vec3& direction,
        const float offset)
    {
        return Ray(point + direction * offset, direction);
    }

    NR_CPU_GPU static Ray invalid()
    {
        return Ray(glm::vec3(0.0f), glm::vec3(0.0f));
    }

private:
    glm::vec3 rayOrigin{};
    glm::vec3 rayDirection{};
};
