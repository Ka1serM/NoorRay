#pragma once

#include <cmath>
#include <cstdint>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "Raytracing/Ray.h"
#include "Samplers/HemisphereSampler.h"
#include "Shading/Spectrum.h"
#include "Samplers/RandomSampler.h"

inline constexpr float LightPi = 3.14159265358979323846f;

NR_CPU_GPU inline float lightSelectionLuminance(const glm::vec3 color)
{
    const float luminance = glm::dot(color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
    return fmaxf(luminance, 0.0f);
}

struct LightSample
{
    glm::vec3 direction{};
    float distance{};
    SampledSpectrum radiance{};
    // Conditional solid-angle PDF for finite-size lights. Zero denotes a
    // delta light. `radiance` is Le/pdf for non-delta samples.
    float pdf{};
};

struct LightHit
{
    float distance{Ray::InfiniteDistance};
    SampledSpectrum radiance{};
    float pdf{};
    uint32_t lightIndex{~uint32_t{0}};
};

NR_CPU_GPU inline void makeLightBasis(
    const glm::vec3 normal, glm::vec3& tangent, glm::vec3& bitangent)
{
    nr::sampling::buildBasis(normal, tangent, bitangent);
}

NR_CPU_GPU inline float oneMinusCosineFromSineSquared(
    const float sineSquared)
{
    const float clamped = fminf(fmaxf(sineSquared, 0.0f), 1.0f);
    // Cycles uses a second-order expansion for small angular extents to avoid
    // cancellation in 1-sqrt(1-sin^2(theta)).
    return clamped > 4.0e-4f
        ? 1.0f - sqrtf(1.0f - clamped)
        : 0.5f * clamped;
}

NR_CPU_GPU inline float oneMinusCosine(const float angle)
{
    return angle > 0.02f ? 1.0f - cosf(angle) : 0.5f * angle * angle;
}

struct UniformConeSample
{
    glm::vec3 direction{};
    float cosine{1.0f};
    float pdf{1.0f};
};

// Cycles' solid-angle cone sampler, including its concentric disk remapping.
// oneMinusCosThetaMax is used instead of cos(theta) so tiny emitters retain
// useful precision.
NR_CPU_GPU inline UniformConeSample sampleUniformCone(
    const glm::vec3 axis,
    const float oneMinusCosThetaMax,
    const glm::vec2 random)
{
    UniformConeSample result{};
    result.direction = axis;
    if (oneMinusCosThetaMax <= 0.0f)
        return result;

    glm::vec2 disk = nr::sampling::concentricDisk(random);
    const float radiusSquared = glm::dot(disk, disk);
    result.cosine = 1.0f - radiusSquared * oneMinusCosThetaMax;
    disk *= sqrtf(fmaxf(oneMinusCosThetaMax
        * (2.0f - oneMinusCosThetaMax * radiusSquared), 0.0f));

    glm::vec3 tangent{}, bitangent{};
    makeLightBasis(axis, tangent, bitangent);
    result.direction = tangent * disk.x + bitangent * disk.y
        + axis * result.cosine;
    result.pdf = 1.0f / (2.0f * LightPi * oneMinusCosThetaMax);
    return result;
}

NR_CPU_GPU inline glm::vec3 sampleUniformSphere(const glm::vec2 random)
{
    const float z = 1.0f - 2.0f * random.x;
    const float radius = sqrtf(fmaxf(1.0f - z * z, 0.0f));
    const float phi = 2.0f * LightPi * random.y;
    return glm::vec3(radius * cosf(phi), radius * sinf(phi), z);
}

NR_CPU_GPU inline bool intersectSphereLight(
    const Ray& ray, const float tMin, const float tMax,
    const glm::vec3 center, const float radius, float& distance)
{
    if (radius <= 0.0f)
        return false;
    const glm::vec3 offset = ray.origin() - center;
    const float a = glm::dot(ray.direction(), ray.direction());
    const float halfB = glm::dot(offset, ray.direction());
    const float c = glm::dot(offset, offset) - radius * radius;
    const float discriminant = halfB * halfB - a * c;
    if (discriminant < 0.0f || a <= 0.0f)
        return false;
    const float root = sqrtf(discriminant);
    const float nearDistance = (-halfB - root) / a;
    const float farDistance = (-halfB + root) / a;
    distance = nearDistance >= tMin ? nearDistance : farDistance;
    return distance >= tMin && distance <= tMax;
}

NR_CPU_GPU inline float sphereLightPdf(
    const glm::vec3 origin, const glm::vec3 center, const float radius)
{
    const float distanceSquared = glm::dot(center - origin, center - origin);
    if (distanceSquared <= radius * radius)
        return 1.0f / (4.0f * LightPi);
    const float sinThetaSquared = fminf(radius * radius / distanceSquared, 1.0f);
    return 1.0f / fmaxf(2.0f * LightPi
        * oneMinusCosineFromSineSquared(sinThetaSquared), 1.0e-20f);
}

NR_CPU_GPU inline LightSample sampleSphereLight(
    const glm::vec3 origin, const glm::vec3 center, const float radius,
    RandomState& rng)
{
    LightSample sample{};
    const glm::vec3 toCenter = center - origin;
    const float centerDistanceSquared = glm::dot(toCenter, toCenter);
    if (radius <= 0.0f)
    {
        sample.distance = sqrtf(centerDistanceSquared);
        if (sample.distance > 0.0f)
            sample.direction = toCenter / sample.distance;
        return sample;
    }

    if (centerDistanceSquared > radius * radius)
    {
        const float centerDistance = sqrtf(centerDistanceSquared);
        const float sinThetaSquared = fminf(
            radius * radius / centerDistanceSquared, 1.0f);
        const UniformConeSample cone = sampleUniformCone(
            toCenter / centerDistance,
            oneMinusCosineFromSineSquared(sinThetaSquared),
            glm::vec2(randomFloat(rng), randomFloat(rng)));
        sample.direction = cone.direction;
        sample.pdf = cone.pdf;

        // Cycles' law-of-cosines intersection avoids a second general sphere
        // intersection and remains stable for very small apparent radii.
        sample.distance = centerDistance * cone.cosine - sqrtf(fmaxf(
            radius * radius - centerDistanceSquared
                + centerDistanceSquared * cone.cosine * cone.cosine,
            0.0f));
        return sample;
    }

    sample.direction = sampleUniformSphere(
        glm::vec2(randomFloat(rng), randomFloat(rng)));

    const Ray sampleRay(origin, sample.direction);
    if (!intersectSphereLight(sampleRay, 0.0f, Ray::InfiniteDistance, center, radius, sample.distance))
        return {};
    sample.pdf = sphereLightPdf(origin, center, radius);
    return sample;
}

NR_CPU_GPU inline float safeArcsine(const float value)
{
    return asinf(fminf(fmaxf(value, -1.0f), 1.0f));
}

NR_CPU_GPU inline float sphericalRectanglePdf(
    const glm::vec3 origin,
    const glm::vec3 center,
    const glm::vec3 sampledPosition,
    const glm::vec3 axisU,
    const float lengthU,
    const glm::vec3 axisV,
    const float lengthV)
{
    const float area = lengthU * lengthV;
    if (lengthU <= 0.0f || lengthV <= 0.0f || area <= 0.0f)
        return 0.0f;

    const glm::vec3 x = axisU;
    const glm::vec3 y = axisV;
    glm::vec3 z = glm::cross(x, y);
    const glm::vec3 direction = center - origin;
    float z0 = glm::dot(direction, z);
    if (z0 > 0.0f)
    {
        z = -z;
        z0 = -z0;
    }

    const float centerX = glm::dot(direction, x);
    const float centerY = glm::dot(direction, y);
    const float x0 = centerX - 0.5f * lengthU;
    const float x1 = centerX + 0.5f * lengthU;
    const float y0 = centerY - 0.5f * lengthV;
    const float y1 = centerY + 0.5f * lengthV;

    float n0 = -y0;
    float n1 = x1;
    float n2 = y1;
    float n3 = -x0;
    n0 /= sqrtf(n0 * n0 + z0 * z0);
    n1 /= sqrtf(n1 * n1 + z0 * z0);
    n2 /= sqrtf(n2 * n2 + z0 * z0);
    n3 /= sqrtf(n3 * n3 + z0 * z0);

    const float solidAngle = -(
        safeArcsine(-n0 * n1)
        + safeArcsine(-n1 * n2)
        + safeArcsine(-n2 * n3)
        + safeArcsine(-n3 * n0));
    const float minimumNormalSquared = fminf(fminf(n0 * n0, n1 * n1),
        fminf(n2 * n2, n3 * n3));
    if (solidAngle >= 1.0e-5f && minimumNormalSquared <= 0.99999f)
        return 1.0f / solidAngle;

    const glm::vec3 sampledDelta = sampledPosition - origin;
    const float distanceSquared = glm::dot(sampledDelta, sampledDelta);
    const float projectedDistance = fabsf(glm::dot(z, sampledDelta));
    return distanceSquared * sqrtf(distanceSquared)
        / fmaxf(projectedDistance * area, 1.0e-20f);
}

// Area-preserving spherical-rectangle sampling from Cycles (Urena et al.).
// The returned PDF is in solid-angle measure and sampledPosition is updated
// only when a numerically valid rectangle can be sampled.
NR_CPU_GPU inline float sampleSphericalRectangle(
    const glm::vec3 origin,
    glm::vec3& sampledPosition,
    const glm::vec3 axisU,
    const float lengthU,
    const glm::vec3 axisV,
    const float lengthV,
    const glm::vec2 random)
{
    const float area = lengthU * lengthV;
    if (lengthU <= 0.0f || lengthV <= 0.0f || area <= 0.0f)
        return 0.0f;

    const glm::vec3 x = axisU;
    const glm::vec3 y = axisV;
    glm::vec3 z = glm::cross(x, y);
    const glm::vec3 direction = sampledPosition - origin;
    float z0 = glm::dot(direction, z);
    if (z0 > 0.0f)
    {
        z = -z;
        z0 = -z0;
    }

    const float centerX = glm::dot(direction, x);
    const float centerY = glm::dot(direction, y);
    const float x0 = centerX - 0.5f * lengthU;
    const float x1 = centerX + 0.5f * lengthU;
    const float y0 = centerY - 0.5f * lengthV;
    const float y1 = centerY + 0.5f * lengthV;

    float n0 = -y0;
    float n1 = x1;
    float n2 = y1;
    float n3 = -x0;
    n0 /= sqrtf(n0 * n0 + z0 * z0);
    n1 /= sqrtf(n1 * n1 + z0 * z0);
    n2 /= sqrtf(n2 * n2 + z0 * z0);
    n3 /= sqrtf(n3 * n3 + z0 * z0);

    const float g0 = safeArcsine(-n0 * n1);
    const float g1 = safeArcsine(-n1 * n2);
    const float g2 = safeArcsine(-n2 * n3);
    const float g3 = safeArcsine(-n3 * n0);
    const float solidAngle = -(g0 + g1 + g2 + g3);

    const float minimumNormalSquared = fminf(fminf(n0 * n0, n1 * n1),
        fminf(n2 * n2, n3 * n3));
    if (solidAngle < 1.0e-5f || minimumNormalSquared > 0.99999f)
    {
        // Cycles falls back to planar area sampling when the solid angle is
        // below single-precision resolution. Use the exact Jacobian at the
        // sampled point instead of the center-point approximation.
        sampledPosition += axisU * ((random.x - 0.5f) * lengthU)
            + axisV * ((random.y - 0.5f) * lengthV);
        const glm::vec3 sampledDelta = sampledPosition - origin;
        const float distanceSquared = glm::dot(sampledDelta, sampledDelta);
        const float projectedDistance = fabsf(glm::dot(z, sampledDelta));
        return distanceSquared * sqrtf(distanceSquared)
            / fmaxf(projectedDistance * area, 1.0e-20f);
    }

    const float b0 = n0;
    const float b1 = n2;
    const float b0Squared = b0 * b0;
    const float angleU = random.x * solidAngle + g2 + g3;
    const float sineU = sinf(angleU);
    const float fU = (cosf(angleU) * b0 + b1)
        / (fabsf(sineU) > 1.0e-20f ? sineU
                                    : copysignf(1.0e-20f, sineU));
    float cosineU = copysignf(1.0f / sqrtf(fU * fU + b0Squared), fU);
    cosineU = fminf(fmaxf(cosineU, -1.0f), 1.0f);
    float sampleX = -(cosineU * z0)
        / fmaxf(sqrtf(fmaxf(1.0f - cosineU * cosineU, 0.0f)), 1.0e-7f);
    sampleX = fminf(fmaxf(sampleX, x0), x1);

    const float distanceSquared = sampleX * sampleX + z0 * z0;
    const float h0 = y0 / sqrtf(distanceSquared + y0 * y0);
    const float h1 = y1 / sqrtf(distanceSquared + y1 * y1);
    const float h = h0 + random.y * (h1 - h0);
    const float hSquared = h * h;
    const float sampleY = hSquared < 1.0f - 1.0e-6f
        ? h * sqrtf(distanceSquared / (1.0f - hSquared)) : y1;

    sampledPosition = origin + sampleX * x + sampleY * y + z0 * z;
    return 1.0f / solidAngle;
}
