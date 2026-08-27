#pragma once

#include <cmath>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "Backend/Host/Platform.h"

namespace nr
{

NR_CPU_GPU inline bool isFinite(const float value)
{
    return std::isfinite(value);
}

NR_CPU_GPU inline bool isFinite(const glm::vec3& value)
{
    return isFinite(value.x) && isFinite(value.y) && isFinite(value.z);
}

NR_CPU_GPU inline glm::vec3 safeNormalize(
    const glm::vec3& value, const glm::vec3& fallback = glm::vec3(0.0f))
{
    const float lengthSquared = glm::dot(value, value);
    return isFinite(lengthSquared) && lengthSquared > 1.0e-12f
        ? value / sqrtf(lengthSquared) : fallback;
}

}

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

    NR_CPU_GPU Ray(
        const glm::vec3& origin,
        const glm::vec3& direction,
        const float minDistance,
        const float maxDistance)
        : rayOrigin(origin),
          rayDirection(direction),
          rayMinDistance(minDistance),
          rayMaxDistance(maxDistance)
    {}

    NR_CPU_GPU const glm::vec3& origin() const { return rayOrigin; }
    NR_CPU_GPU const glm::vec3& direction() const { return rayDirection; }
    NR_CPU_GPU float minDistance() const { return rayMinDistance; }
    NR_CPU_GPU float maxDistance() const { return rayMaxDistance; }
    NR_CPU_GPU bool hasTraversalInterval() const
    {
        return isValid() && rayMinDistance <= rayMaxDistance;
    }

    NR_CPU_GPU bool isFinite() const
    {
        return nr::isFinite(rayOrigin) && nr::isFinite(rayDirection);
    }

    NR_CPU_GPU bool isValid() const
    {
        const float directionLengthSquared = glm::dot(rayDirection, rayDirection);
        return isFinite() && nr::isFinite(directionLengthSquared)
            && directionLengthSquared > 1.0e-12f;
    }

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

    NR_CPU_GPU Ray withMinDistance(const float minDistance) const
    {
        return Ray(rayOrigin, rayDirection, minDistance, rayMaxDistance);
    }

    NR_CPU_GPU Ray withMaxDistance(const float maxDistance) const
    {
        return Ray(rayOrigin, rayDirection, rayMinDistance, maxDistance);
    }

    NR_CPU_GPU static Ray invalid()
    {
        return Ray(glm::vec3(0.0f), glm::vec3(0.0f));
    }

private:
    glm::vec3 rayOrigin{};
    glm::vec3 rayDirection{};
    float rayMinDistance{0.0f};
    float rayMaxDistance{InfiniteDistance};
};
