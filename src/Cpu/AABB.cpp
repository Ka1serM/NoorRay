#include "AABB.h"
#include <algorithm>
#include <limits>

AABB::AABB() : min(std::numeric_limits<float>::max()), max(std::numeric_limits<float>::lowest()) {}

void AABB::expand(const glm::vec3& point) {
    min = glm::min(min, point);
    max = glm::max(max, point);
}

void AABB::expand(const AABB& other) {
    min = glm::min(min, other.min);
    max = glm::max(max, other.max);
}

float AABB::surfaceArea() const {
    glm::vec3 extent = max - min;
    // Handle degenerate (flat) boxes
    if (extent.x < 0 || extent.y < 0 || extent.z < 0) return 0.0f;
    return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
}

bool AABB::intersect(const glm::vec3& origin, const glm::vec3& direction, float tMin, float tMax) const {
    glm::vec3 invDir = 1.0f / direction;
    glm::vec3 t1 = (min - origin) * invDir;
    glm::vec3 t2 = (max - origin) * invDir;

    glm::vec3 tNear = glm::min(t1, t2);
    glm::vec3 tFar = glm::max(t1, t2);

    float tEnter = glm::max(glm::max(tNear.x, tNear.y), tNear.z);
    float tExit = glm::min(glm::min(tFar.x, tFar.y), tFar.z);

    // The interval [tEnter, tExit] must overlap with the ray's interval [tMin, tMax]
    return tExit >= tEnter && tExit >= tMin && tEnter < tMax;
}