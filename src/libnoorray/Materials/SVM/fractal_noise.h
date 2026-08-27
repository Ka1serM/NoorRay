/* SPDX-License-Identifier: Apache-2.0
 *
 * MaterialX fractal-noise evaluation. The octave loop follows the standard
 * fractal3d reference implementation.
 */

#pragma once

#include <glm/common.hpp>

#include "Backend/Host/Platform.h"

namespace nr::svm::detail
{
NR_GPU inline std::uint32_t fractalRotl32(const std::uint32_t x, const unsigned int k)
{
    return (x << k) | (x >> (32 - k));
}
NR_GPU inline std::uint32_t fractalBjfinal(std::uint32_t a, std::uint32_t b, std::uint32_t c)
{
    c ^= b; c -= fractalRotl32(b, 14); a ^= c; a -= fractalRotl32(c, 11);
    b ^= a; b -= fractalRotl32(a, 25); c ^= b; c -= fractalRotl32(b, 16);
    a ^= c; a -= fractalRotl32(c, 4); b ^= a; b -= fractalRotl32(a, 14);
    c ^= b; c -= fractalRotl32(b, 24); return c;
}
NR_GPU inline std::uint32_t fractalHashInt3(const int x, const int y, const int z)
{
    constexpr std::uint32_t seed = 0xdeadbeefu + (3u << 2u) + 13u;
    return fractalBjfinal(seed + static_cast<std::uint32_t>(x),
        seed + static_cast<std::uint32_t>(y), seed + static_cast<std::uint32_t>(z));
}
NR_GPU inline std::uint32_t fractalHashInt2(const int x, const int y)
{
    constexpr std::uint32_t seed = 0xdeadbeefu + (2u << 2u) + 13u;
    return fractalBjfinal(seed + static_cast<std::uint32_t>(x),
        seed + static_cast<std::uint32_t>(y), seed);
}
NR_GPU inline float fractalFade(const float t)
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}
NR_GPU inline float fractalGradient(const std::uint32_t hash, const float x, const float y,
    const float z)
{
    const std::uint32_t h = hash & 15u;
    const float u = h < 8u ? x : y;
    const float v = h < 4u ? y : ((h == 12u || h == 14u) ? x : z);
    return ((h & 1u) ? -u : u) + ((h & 2u) ? -v : v);
}
NR_GPU inline float svmPerlin3d(const glm::vec3 p)
{
    const float floorX = floorf(p.x), floorY = floorf(p.y), floorZ = floorf(p.z);
    const int x = static_cast<int>(floorX), y = static_cast<int>(floorY);
    const int z = static_cast<int>(floorZ);
    const float fx = p.x - floorX, fy = p.y - floorY, fz = p.z - floorZ;
    const float u = fractalFade(fx), v = fractalFade(fy), w = fractalFade(fz);
    const float n000 = fractalGradient(fractalHashInt3(x, y, z), fx, fy, fz);
    const float n100 = fractalGradient(fractalHashInt3(x + 1, y, z), fx - 1.0f, fy, fz);
    const float n010 = fractalGradient(fractalHashInt3(x, y + 1, z), fx, fy - 1.0f, fz);
    const float n110 = fractalGradient(fractalHashInt3(x + 1, y + 1, z), fx - 1.0f, fy - 1.0f, fz);
    const float n001 = fractalGradient(fractalHashInt3(x, y, z + 1), fx, fy, fz - 1.0f);
    const float n101 = fractalGradient(fractalHashInt3(x + 1, y, z + 1), fx - 1.0f, fy, fz - 1.0f);
    const float n011 = fractalGradient(fractalHashInt3(x, y + 1, z + 1), fx, fy - 1.0f, fz - 1.0f);
    const float n111 = fractalGradient(fractalHashInt3(x + 1, y + 1, z + 1), fx - 1.0f, fy - 1.0f, fz - 1.0f);
    const float z0 = glm::mix(glm::mix(n000, n100, u), glm::mix(n010, n110, u), v);
    const float z1 = glm::mix(glm::mix(n001, n101, u), glm::mix(n011, n111, u), v);
    return 0.9820f * glm::mix(z0, z1, w);
}
NR_GPU inline float fractalGradient2d(const std::uint32_t hash, const float x,
    const float y)
{
    const std::uint32_t h = hash & 7u;
    const float u = h < 4u ? x : y;
    const float v = 2.0f * (h < 4u ? y : x);
    return ((h & 1u) ? -u : u) + ((h & 2u) ? -v : v);
}
NR_GPU inline float svmPerlin2d(const glm::vec2 p)
{
    const float floorX = floorf(p.x), floorY = floorf(p.y);
    const int x = static_cast<int>(floorX);
    const int y = static_cast<int>(floorY);
    const float fx = p.x - floorX, fy = p.y - floorY;
    const float u = fractalFade(fx), v = fractalFade(fy);
    const float n00 = fractalGradient2d(fractalHashInt2(x, y), fx, fy);
    const float n10 = fractalGradient2d(fractalHashInt2(x + 1, y), fx - 1.0f, fy);
    const float n01 = fractalGradient2d(fractalHashInt2(x, y + 1), fx, fy - 1.0f);
    const float n11 = fractalGradient2d(fractalHashInt2(x + 1, y + 1), fx - 1.0f, fy - 1.0f);
    return 0.6616f * glm::mix(glm::mix(n00, n10, u), glm::mix(n01, n11, u), v);
}
NR_GPU inline glm::vec3 svmPerlin3dVec3(const glm::vec3 p)
{
    const float floorX = floorf(p.x), floorY = floorf(p.y), floorZ = floorf(p.z);
    const int x = static_cast<int>(floorX), y = static_cast<int>(floorY);
    const int z = static_cast<int>(floorZ);
    const float fx = p.x - floorX, fy = p.y - floorY, fz = p.z - floorZ;
    const float u = fractalFade(fx), v = fractalFade(fy), w = fractalFade(fz);
    const auto gradient = [](const int ix, const int iy, const int iz, const float px, const float py, const float pz) {
        const std::uint32_t h = fractalHashInt3(ix, iy, iz);
        return glm::vec3(fractalGradient(h & 0xffu, px, py, pz),
            fractalGradient((h >> 8u) & 0xffu, px, py, pz),
            fractalGradient((h >> 16u) & 0xffu, px, py, pz));
    };
    const glm::vec3 n000 = gradient(x, y, z, fx, fy, fz);
    const glm::vec3 n100 = gradient(x + 1, y, z, fx - 1.0f, fy, fz);
    const glm::vec3 n010 = gradient(x, y + 1, z, fx, fy - 1.0f, fz);
    const glm::vec3 n110 = gradient(x + 1, y + 1, z, fx - 1.0f, fy - 1.0f, fz);
    const glm::vec3 n001 = gradient(x, y, z + 1, fx, fy, fz - 1.0f);
    const glm::vec3 n101 = gradient(x + 1, y, z + 1, fx - 1.0f, fy, fz - 1.0f);
    const glm::vec3 n011 = gradient(x, y + 1, z + 1, fx, fy - 1.0f, fz - 1.0f);
    const glm::vec3 n111 = gradient(x + 1, y + 1, z + 1, fx - 1.0f, fy - 1.0f, fz - 1.0f);
    return 0.9820f * glm::mix(glm::mix(glm::mix(n000, n100, u), glm::mix(n010, n110, u), v),
        glm::mix(glm::mix(n001, n101, u), glm::mix(n011, n111, u), v), w);
}
NR_GPU inline glm::vec3 svmPerlin2dVec3(const glm::vec2 p)
{
    const float floorX = floorf(p.x), floorY = floorf(p.y);
    const int x = static_cast<int>(floorX);
    const int y = static_cast<int>(floorY);
    const float fx = p.x - floorX, fy = p.y - floorY;
    const float u = fractalFade(fx), v = fractalFade(fy);
    const auto gradient = [](const int ix, const int iy, const float px, const float py) {
        const std::uint32_t h = fractalHashInt2(ix, iy);
        return glm::vec3(fractalGradient2d(h & 0xffu, px, py),
            fractalGradient2d((h >> 8u) & 0xffu, px, py),
            fractalGradient2d((h >> 16u) & 0xffu, px, py));
    };
    const glm::vec3 n00 = gradient(x, y, fx, fy);
    const glm::vec3 n10 = gradient(x + 1, y, fx - 1.0f, fy);
    const glm::vec3 n01 = gradient(x, y + 1, fx, fy - 1.0f);
    const glm::vec3 n11 = gradient(x + 1, y + 1, fx - 1.0f, fy - 1.0f);
    return 0.6616f * glm::mix(glm::mix(n00, n10, u), glm::mix(n01, n11, u), v);
}
NR_GPU inline float svmFractal3d(glm::vec3 p, const int octaves, const float lacunarity,
    const float diminish)
{
    float result = 0.0f, amplitude = 1.0f;
    for (int i = 0; i < octaves; ++i) {
        result += amplitude * svmPerlin3d(p);
        amplitude *= diminish;
        p *= lacunarity;
    }
    return result;
}
NR_GPU inline glm::vec3 svmFractal3dVec3(glm::vec3 p, const int octaves,
    const float lacunarity, const float diminish)
{
    glm::vec3 result(0.0f);
    float amplitude = 1.0f;
    for (int i = 0; i < octaves; ++i) {
        result += amplitude * svmPerlin3dVec3(p);
        amplitude *= diminish;
        p *= lacunarity;
    }
    return result;
}
NR_GPU inline float svmFractal2d(glm::vec2 p, const int octaves,
    const float lacunarity, const float diminish)
{
    float result = 0.0f, amplitude = 1.0f;
    for (int i = 0; i < octaves; ++i) {
        result += amplitude * svmPerlin2d(p);
        amplitude *= diminish;
        p *= lacunarity;
    }
    return result;
}
NR_GPU inline glm::vec3 svmFractal2dVec3(glm::vec2 p, const int octaves,
    const float lacunarity, const float diminish)
{
    glm::vec3 result(0.0f);
    float amplitude = 1.0f;
    for (int i = 0; i < octaves; ++i) {
        result += amplitude * svmPerlin2dVec3(p);
        amplitude *= diminish;
        p *= lacunarity;
    }
    return result;
}
} // namespace nr::svm::detail
