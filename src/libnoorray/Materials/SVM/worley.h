/* SPDX-License-Identifier: Apache-2.0
 *
 * MaterialX mx_worleynoise3d port.  This is kept as a dedicated fixed SVM
 * operation with a fixed SVM payload.
 */
#pragma once

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include "Backend/CUDA/Annotations.h"
#include "Materials/SVM/fractal_noise.h"

namespace nr::svm::detail
{
NR_GPU inline float svmCellNoise3d(const glm::vec3 p)
{
    return static_cast<float>(fractalHashInt3(static_cast<int>(floorf(p.x)),
        static_cast<int>(floorf(p.y)), static_cast<int>(floorf(p.z)))) / 4294967295.0f;
}
NR_GPU inline float svmCellNoise2d(const glm::vec3 p)
{
    return static_cast<float>(fractalHashInt2(static_cast<int>(floorf(p.x)),
        static_cast<int>(floorf(p.y)))) / 4294967295.0f;
}
NR_GPU inline glm::vec3 svmCellNoise3dVec3(const glm::vec3 p)
{
    const std::uint32_t h = fractalHashInt3(static_cast<int>(floorf(p.x)),
        static_cast<int>(floorf(p.y)), static_cast<int>(floorf(p.z)));
    return glm::vec3(static_cast<float>(h & 0xffu), static_cast<float>((h >> 8u) & 0xffu),
        static_cast<float>((h >> 16u) & 0xffu)) / 255.0f;
}
NR_GPU inline glm::vec3 svmCellNoise2dVec3(const glm::vec2 p)
{
    const std::uint32_t h = fractalHashInt2(static_cast<int>(floorf(p.x)),
        static_cast<int>(floorf(p.y)));
    return glm::vec3(static_cast<float>(h & 0xffu), static_cast<float>((h >> 8u) & 0xffu),
        static_cast<float>((h >> 16u) & 0xffu)) / 255.0f;
}
NR_GPU inline glm::vec3 svmWorleyCellPosition3d(const int x, const int y, const int z,
    const int xoff, const int yoff, const int zoff, const float jitter)
{
    glm::vec3 off = svmCellNoise3dVec3(glm::vec3(x + xoff, y + yoff, z + zoff));
    off = (off - glm::vec3(0.5f)) * jitter + glm::vec3(0.5f);
    return glm::vec3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)) + off;
}
NR_GPU inline glm::vec3 svmWorleyNoise3d(const glm::vec3 p, const float jitter,
    const int style, const bool vectorResult)
{
    const int X = static_cast<int>(floorf(p.x));
    const int Y = static_cast<int>(floorf(p.y));
    const int Z = static_cast<int>(floorf(p.z));
    const glm::vec3 local = p - glm::vec3(
        static_cast<float>(X), static_cast<float>(Y), static_cast<float>(Z));
    glm::vec3 nearest(1.0e6f);
    glm::vec3 minPos(0.0f);
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
            for (int z = -1; z <= 1; ++z) {
                const glm::vec3 cell = svmWorleyCellPosition3d(x, y, z, X, Y, Z, jitter) - local;
                const float distance = glm::dot(cell, cell);
                if (distance < nearest.x) {
                    nearest.z = nearest.y; nearest.y = nearest.x; nearest.x = distance; minPos = cell;
                }
                else if (distance < nearest.y) { nearest.z = nearest.y; nearest.y = distance; }
                else if (distance < nearest.z) nearest.z = distance;
            }
    if (style == 1)
        return vectorResult ? svmCellNoise3dVec3(minPos + p) : glm::vec3(svmCellNoise3d(minPos + p));
    nearest = glm::sqrt(nearest);
    return vectorResult ? nearest : glm::vec3(nearest.x);
}
NR_GPU inline glm::vec3 svmWorleyNoise2d(const glm::vec2 p, const float jitter,
    const int style, const bool vectorResult)
{
    const int X = static_cast<int>(floorf(p.x));
    const int Y = static_cast<int>(floorf(p.y));
    const glm::vec2 local = p - glm::vec2(
        static_cast<float>(X), static_cast<float>(Y));
    glm::vec3 nearest(1.0e6f);
    glm::vec2 minPos(0.0f);
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y) {
            const glm::vec3 random = svmCellNoise2dVec3(glm::vec2(X + x, Y + y));
            const glm::vec2 cell = glm::vec2(static_cast<float>(x), static_cast<float>(y))
                + (glm::vec2(random) - glm::vec2(0.5f)) * jitter + glm::vec2(0.5f);
            const glm::vec2 diff = cell - local;
            const float distance = glm::dot(diff, diff);
            if (distance < nearest.x) {
                nearest.z = nearest.y; nearest.y = nearest.x; nearest.x = distance; minPos = diff;
            }
            else if (distance < nearest.y) { nearest.z = nearest.y; nearest.y = distance; }
            else if (distance < nearest.z) nearest.z = distance;
        }
    if (style == 1)
        return vectorResult ? svmCellNoise2dVec3(minPos + p) : glm::vec3(svmCellNoise2d(glm::vec3(minPos + p, 0.0f)));
    nearest = glm::sqrt(nearest);
    return vectorResult ? nearest : glm::vec3(nearest.x);
}
} // namespace nr::svm::detail
