#include "../../Shaders/PathTracing/SharedStructs.h"
#include <algorithm>
#include <limits>

#include "BVH.h"
#include "glm/common.hpp"

//Definition of AABB Struct is in Shaders/SharedStructs.h

AABB::AABB() : min(std::numeric_limits<float>::max()), max(std::numeric_limits<float>::lowest())
{
}

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

bool AABB::intersect(const glm::vec3& origin, const glm::vec3& invDir, const glm::ivec3& dirIsNeg, float& tNear, float& tFar) const {
    vec3 t1 = (min - origin) * invDir;
    vec3 t2 = (max - origin) * invDir;
    
    if (dirIsNeg.x)
        std::swap(t1.x, t2.x);
    if (dirIsNeg.y)
        std::swap(t1.y, t2.y);
    if (dirIsNeg.z)
        std::swap(t1.z, t2.z);
    
    tNear = std::max({t1.x, t1.y, t1.z});
    tFar = std::min({t2.x, t2.y, t2.z});
    
    return tNear <= tFar;
}