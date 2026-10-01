#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/matrix.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

// NoorRay's world is Unreal's: centimetres, Z up, left-handed (X forward, Y right).
// Importers convert their source's space into it with these helpers.
namespace nr::coords {

struct CoordinateSpace {
    // The source's axes in NoorRay's world.
    glm::vec3 xAxis{1.0f, 0.0f, 0.0f};
    glm::vec3 yAxis{0.0f, 1.0f, 0.0f};
    glm::vec3 zAxis{0.0f, 0.0f, 1.0f};
    float centimetresPerUnit = 1.0f;
};

// glTF, OBJ and OpenGL: metres, Y up, right-handed.
inline constexpr CoordinateSpace OpenGlSpace{
    {1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 0.0f},
    100.0f,
};

// OpenCV and COLMAP: metres, Y down, Z forward, right-handed.
inline constexpr CoordinateSpace YDownZForwardSpace{
    {1.0f,  0.0f,  0.0f},
    {0.0f,  0.0f, -1.0f},
    {0.0f, -1.0f,  0.0f},
    100.0f,
};

// Blender and 3ds Max: metres, Z up, Y forward, right-handed.
inline constexpr CoordinateSpace ZUpYForwardSpace{
    {1.0f,  0.0f, 0.0f},
    {0.0f, -1.0f, 0.0f},
    {0.0f,  0.0f, 1.0f},
    100.0f,
};

inline glm::mat3 axes(const CoordinateSpace& space)
{
    return {space.xAxis, space.yAxis, space.zAxis};
}

// A right-handed source mirrors into the left-handed world: its triangles' winding
// and tangent frames' handedness reverse.
inline bool isMirrored(const CoordinateSpace& space)
{
    return glm::determinant(axes(space)) < 0.0f;
}

inline glm::vec3 toWorldPosition(const glm::vec3 p, const CoordinateSpace& space)
{
    return axes(space) * p * space.centimetresPerUnit;
}

inline glm::vec3 toWorldDirection(const glm::vec3 d, const CoordinateSpace& space)
{
    return axes(space) * d;
}

// A source-space transform as the equivalent world transform.
inline glm::mat4 toWorldTransform(const glm::mat4& sourceTransform, const CoordinateSpace& space)
{
    const glm::mat4 conversion(glm::vec4(space.xAxis * space.centimetresPerUnit, 0.0f),
                               glm::vec4(space.yAxis * space.centimetresPerUnit, 0.0f),
                               glm::vec4(space.zAxis * space.centimetresPerUnit, 0.0f),
                               glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    return conversion * sourceTransform * glm::inverse(conversion);
}

// Keeps triangles facing the same way after a mirroring conversion.
inline void reverseWinding(std::vector<uint32_t>& indices)
{
    for (size_t triangle = 0; triangle + 2 < indices.size(); triangle += 3)
        std::swap(indices[triangle + 1], indices[triangle + 2]);
}

} // namespace nr::coords
