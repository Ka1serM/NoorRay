// Native sequential realistic-lens support.  This is deliberately a POD-only
// ABI: the same records can be copied to any GPU backend without exposing a
// parser or a third-party optical library to shaders.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

#include "Backend/Host/Platform.h"
#include "Materials/Shading/Sellmeier.h"

namespace nr::optics {

constexpr uint32_t MaxLensSurfaces = 64;
constexpr uint32_t MaxAsphereTerms = 8;

struct Medium {
    SellmeierCoefficients sellmeier{};
};

struct Surface {
    // All distances are millimetres in lens coordinates, with film at z = 0.
    float z{};
    float radius{};             // 0 denotes a plane.
    float apertureRadius{};
    float conic{};
    std::array<float, MaxAsphereTerms> asphere{}; // r^4 ... r^18
    uint32_t mediumAfter{};     // medium reached travelling from film to world.
    uint32_t isStop{};
};

struct LensSnapshot {
    std::array<Surface, MaxLensSurfaces> surfaces{};
    std::array<Medium, MaxLensSurfaces + 1> media{};
    uint32_t surfaceCount{};
    uint32_t mediumCount{1};
    float rearPupilZ{};
    float rearPupilRadius{};
    float focalLengthMm{};
    float sensorWidthMm{};
    float sensorHeightMm{};
};

// Keep the fixed offsets consumed by the Vulkan Slang lens tracer explicit.
// A layout change must update both the host upload and the shader decoder.
static_assert(sizeof(Surface) == 56);
static_assert(sizeof(Medium) == 24);
static_assert(offsetof(LensSnapshot, surfaceCount) == 5144);
static_assert(offsetof(LensSnapshot, rearPupilZ) == 5152);
static_assert(offsetof(LensSnapshot, sensorWidthMm) == 5164);

struct ParsedLens {
    LensSnapshot snapshot{};
    std::string message;
};

// Parsing is intentionally host-only.  Errors include a source path and line
// number; accepting unknown optical commands would silently change a lens.
ParsedLens loadZmx(const std::string& lensPath,
    const std::vector<std::string>& catalogPaths);
std::vector<std::string> splitCatalogPaths(const std::string& paths);

NR_CPU_GPU inline float sag(const Surface& s, const float r2)
{
    float value = 0.f;
    if (fabsf(s.radius) > 1e-6f) {
        const float c = 1.f / s.radius;
        const float root = fmaxf(0.f, 1.f - (1.f + s.conic) * c * c * r2);
        value = c * r2 / (1.f + sqrtf(root));
    }
    float power = r2 * r2;
    for (uint32_t i = 0; i < MaxAsphereTerms; ++i) {
        value += s.asphere[i] * power;
        power *= r2;
    }
    return value;
}

NR_CPU_GPU inline glm::vec3 surfaceNormal(const Surface& s, const glm::vec3& p)
{
    const float r2 = p.x * p.x + p.y * p.y;
    float derivative = 0.f;
    if (fabsf(s.radius) > 1e-6f) {
        const float c = 1.f / s.radius;
        const float root = sqrtf(fmaxf(1e-12f, 1.f - (1.f + s.conic) * c * c * r2));
        derivative = c / root;
    }
    float power = r2;
    for (uint32_t i = 0; i < MaxAsphereTerms; ++i) {
        derivative += 2.f * static_cast<float>(i + 2) * s.asphere[i] * power;
        power *= r2;
    }
    return glm::normalize(glm::vec3(-derivative * p.x, -derivative * p.y, 1.f));
}

NR_CPU_GPU inline bool intersect(const Surface& s, const glm::vec3& o,
    const glm::vec3& d, glm::vec3& hit)
{
    float t = (s.z - o.z) / d.z;
    if (!(t > 0.f)) return false;
    // Newton is robust for the modest aspheres represented by the ZMX subset.
    for (int i = 0; i < 12; ++i) {
        const glm::vec3 p = o + t * d;
        const float r2 = p.x * p.x + p.y * p.y;
        const float f = p.z - s.z - sag(s, r2);
        const glm::vec3 n = surfaceNormal(s, p);
        const float df = glm::dot(n, d) / n.z;
        if (fabsf(df) < 1e-7f) return false;
        t -= f / df;
        if (fabsf(f) < 1e-5f) break;
    }
    if (!(t > 0.f)) return false;
    hit = o + t * d;
    return hit.x * hit.x + hit.y * hit.y <= s.apertureRadius * s.apertureRadius;
}

NR_CPU_GPU inline bool refract(const glm::vec3& incident, glm::vec3 normal,
    const float etaI, const float etaT, glm::vec3& transmitted)
{
    if (glm::dot(incident, normal) > 0.f) normal = -normal;
    const float eta = etaI / etaT;
    const float cosI = -glm::dot(normal, incident);
    const float sinT2 = eta * eta * fmaxf(0.f, 1.f - cosI * cosI);
    if (sinT2 >= 1.f) return false;
    transmitted = glm::normalize(eta * incident + (eta * cosI - sqrtf(1.f - sinT2)) * normal);
    return true;
}

// Traces from the film through surfaces ordered from the rear of the lens to
// the world.  The caller samples the rear exit-pupil disk before calling it.
NR_CPU_GPU inline bool traceFilmToWorld(const LensSnapshot& lens,
    glm::vec3& origin, glm::vec3& direction, const float wavelengthNm)
{
    uint32_t medium = 0;
    for (uint32_t i = 0; i < lens.surfaceCount; ++i) {
        const Surface& s = lens.surfaces[i];
        glm::vec3 hit;
        if (!intersect(s, origin, direction, hit)) return false;
        glm::vec3 next;
        const float etaI = sellmeierIor(lens.media[medium].sellmeier, wavelengthNm);
        const uint32_t nextMedium = s.mediumAfter < lens.mediumCount ? s.mediumAfter : 0;
        const float etaT = sellmeierIor(lens.media[nextMedium].sellmeier, wavelengthNm);
        if (!refract(direction, surfaceNormal(s, hit), etaI, etaT, next)) return false;
        origin = hit;
        direction = next;
        medium = nextMedium;
    }
    return true;
}

} // namespace nr::optics
