#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace nr::coords {

struct CoordinateSpace {
    glm::vec3 xAxis{1.0f, 0.0f, 0.0f};
    glm::vec3 yAxis{0.0f, 1.0f, 0.0f};
    glm::vec3 zAxis{0.0f, 0.0f, 1.0f};
};

inline constexpr CoordinateSpace OpenGlSpace{
    {1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 1.0f},
};

inline constexpr CoordinateSpace YDownZForwardSpace{
    {1.0f,  0.0f,  0.0f},
    {0.0f, -1.0f,  0.0f},
    {0.0f,  0.0f, -1.0f},
};

inline constexpr CoordinateSpace ZUpYForwardSpace{
    {1.0f, 0.0f,  0.0f},
    {0.0f, 0.0f, -1.0f},
    {0.0f, 1.0f,  0.0f},
};

inline glm::mat4 toOpenGlMatrix(const CoordinateSpace& sourceSpace)
{
    return {
        sourceSpace.xAxis.x, sourceSpace.xAxis.y, sourceSpace.xAxis.z, 0.0f,
        sourceSpace.yAxis.x, sourceSpace.yAxis.y, sourceSpace.yAxis.z, 0.0f,
        sourceSpace.zAxis.x, sourceSpace.zAxis.y, sourceSpace.zAxis.z, 0.0f,
        0.0f,               0.0f,               0.0f,               1.0f,
    };
}

inline glm::vec3 toOpenGlVector(const glm::vec3 v, const CoordinateSpace& sourceSpace)
{
    return glm::vec3(toOpenGlMatrix(sourceSpace) * glm::vec4(v, 0.0f));
}

inline glm::mat4 toOpenGlTransform(const glm::mat4& sourceTransform, const CoordinateSpace& sourceSpace)
{
    const glm::mat4 conversion = toOpenGlMatrix(sourceSpace);
    return conversion * sourceTransform * inverse(conversion);
}

} // namespace nr::coords
