#pragma once

#include "Sellmeier.slang"

// Scalar decoder for nr::optics::LensSnapshot. The host ABI is asserted in
// KolbLens.h; keeping this pointer-only avoids shader bindings and descriptors.
static const uint LensSurfaceStride = 56u;
static const uint LensMediaOffset = 3584u;
static const uint LensMediumStride = 24u;
static const uint LensSurfaceCountOffset = 5144u;
static const uint LensMediumCountOffset = 5148u;
static const uint LensRearPupilZOffset = 5152u;
static const uint LensRearPupilRadiusOffset = 5156u;

float lensFloat(ByteAddressBuffer lens, uint offset)
{
    return asfloat(lens.Load(offset));
}

float lensSag(ByteAddressBuffer lens, uint surface, float radiusSquared)
{
    const uint base = surface * LensSurfaceStride;
    const float radius = lensFloat(lens, base + 4u);
    const float conic = lensFloat(lens, base + 12u);
    float value = 0.0;
    if (abs(radius) > 1.0e-6)
    {
        const float curvature = 1.0 / radius;
        const float root = max(0.0, 1.0 - (1.0 + conic)
            * curvature * curvature * radiusSquared);
        value = curvature * radiusSquared / (1.0 + sqrt(root));
    }
    float power = radiusSquared * radiusSquared;
    [unroll]
    for (uint i = 0u; i < 8u; ++i)
    {
        value += lensFloat(lens, base + 16u + i * 4u) * power;
        power *= radiusSquared;
    }
    return value;
}

float3 lensSurfaceNormal(ByteAddressBuffer lens, uint surface, float3 point)
{
    const uint base = surface * LensSurfaceStride;
    const float radiusSquared = dot(point.xy, point.xy);
    const float radius = lensFloat(lens, base + 4u);
    const float conic = lensFloat(lens, base + 12u);
    float derivative = 0.0;
    if (abs(radius) > 1.0e-6)
    {
        const float curvature = 1.0 / radius;
        const float root = sqrt(max(1.0e-12, 1.0 - (1.0 + conic)
            * curvature * curvature * radiusSquared));
        derivative = curvature / root;
    }
    float power = radiusSquared;
    [unroll]
    for (uint i = 0u; i < 8u; ++i)
    {
        derivative += 2.0 * float(i + 2u)
            * lensFloat(lens, base + 16u + i * 4u) * power;
        power *= radiusSquared;
    }
    return normalize(float3(-derivative * point.x,
        -derivative * point.y, 1.0));
}

bool lensIntersect(ByteAddressBuffer lens, uint surface, float3 origin,
    float3 direction, out float3 hit)
{
    hit = float3(0.0);
    const uint base = surface * LensSurfaceStride;
    const float surfaceZ = lensFloat(lens, base);
    float t = (surfaceZ - origin.z) / direction.z;
    if (!(t > 0.0))
        return false;
    [loop]
    for (uint iteration = 0u; iteration < 12u; ++iteration)
    {
        const float3 point = origin + t * direction;
        const float f = point.z - surfaceZ
            - lensSag(lens, surface, dot(point.xy, point.xy));
        const float3 normal = lensSurfaceNormal(lens, surface, point);
        const float derivative = dot(normal, direction) / normal.z;
        if (abs(derivative) < 1.0e-7)
            return false;
        t -= f / derivative;
        if (abs(f) < 1.0e-5)
            break;
    }
    if (!(t > 0.0))
        return false;
    hit = origin + t * direction;
    const float apertureRadius = lensFloat(lens, base + 8u);
    return dot(hit.xy, hit.xy) <= apertureRadius * apertureRadius;
}

float lensMediumIor(ByteAddressBuffer lens, uint medium, float wavelengthNm)
{
    const uint base = LensMediaOffset + medium * LensMediumStride;
    return sellmeierIor(float3(lensFloat(lens, base),
        lensFloat(lens, base + 4u), lensFloat(lens, base + 8u)),
        float3(lensFloat(lens, base + 12u), lensFloat(lens, base + 16u),
            lensFloat(lens, base + 20u)), wavelengthNm);
}

bool lensRefract(float3 incident, float3 normal, float etaIncident,
    float etaTransmitted, out float3 transmitted)
{
    transmitted = float3(0.0);
    if (dot(incident, normal) > 0.0)
        normal = -normal;
    const float eta = etaIncident / etaTransmitted;
    const float cosineIncident = -dot(normal, incident);
    const float sineTransmittedSquared = eta * eta
        * max(0.0, 1.0 - cosineIncident * cosineIncident);
    if (sineTransmittedSquared >= 1.0)
        return false;
    transmitted = normalize(eta * incident + (eta * cosineIncident
        - sqrt(1.0 - sineTransmittedSquared)) * normal);
    return true;
}

bool traceLensFilmToWorld(ByteAddressBuffer lens, inout float3 origin,
    inout float3 direction, float wavelengthNm)
{
    const uint surfaceCount = min(lens.Load(LensSurfaceCountOffset), 64u);
    const uint mediumCount = max(min(lens.Load(LensMediumCountOffset), 65u), 1u);
    uint medium = 0u;
    [loop]
    for (uint surface = 0u; surface < surfaceCount; ++surface)
    {
        float3 hit;
        if (!lensIntersect(lens, surface, origin, direction, hit))
            return false;
        const uint base = surface * LensSurfaceStride;
        const uint nextMedium = min(lens.Load(base + 48u), mediumCount - 1u);
        float3 nextDirection;
        if (!lensRefract(direction, lensSurfaceNormal(lens, surface, hit),
                lensMediumIor(lens, medium, wavelengthNm),
                lensMediumIor(lens, nextMedium, wavelengthNm), nextDirection))
            return false;
        origin = hit;
        direction = nextDirection;
        medium = nextMedium;
    }
    return surfaceCount != 0u;
}
