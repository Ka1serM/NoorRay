#pragma once
#include <glm/glm.hpp>

// Axis-Aligned Bounding Box
struct AABB {
    glm::vec3 min, max;

    AABB();

    void expand(const glm::vec3& point);
    void expand(const AABB& other);
    float surfaceArea() const;
    bool intersect(const glm::vec3& origin, const glm::vec3& direction, float tMin, float tMax) const;
};