#pragma once

#include "../../Shared/Lens.h"
#include "Sellmeier.slang"

float lensSag(Lens* lens, uint surfaceIndex, float radiusSquared)
{
    const Surface surface = lens->surfaces[surfaceIndex];
    float value = 0.0;
    if (abs(surface.radius) > 1.0e-6)
    {
        const float curvature = 1.0 / surface.radius;
        const float root = max(0.0, 1.0 - (1.0 + surface.conic)
            * curvature * curvature * radiusSquared);
        value = curvature * radiusSquared / (1.0 + sqrt(root));
    }
    float power = radiusSquared * radiusSquared;
    [unroll]
    for (uint i = 0u; i < MaxAsphereTerms; ++i)
    {
        value += surface.asphere[i] * power;
        power *= radiusSquared;
    }
    return value;
}

float3 lensSurfaceNormal(Lens* lens, uint surfaceIndex, float3 point)
{
    const Surface surface = lens->surfaces[surfaceIndex];
    const float radiusSquared = dot(point.xy, point.xy);
    float derivative = 0.0;
    if (abs(surface.radius) > 1.0e-6)
    {
        const float curvature = 1.0 / surface.radius;
        const float root = sqrt(max(1.0e-12, 1.0 - (1.0 + surface.conic)
            * curvature * curvature * radiusSquared));
        derivative = curvature / root;
    }
    float power = radiusSquared;
    [unroll]
    for (uint i = 0u; i < MaxAsphereTerms; ++i)
    {
        derivative += 2.0 * float(i + 2u) * surface.asphere[i] * power;
        power *= radiusSquared;
    }
    return normalize(float3(-derivative * point.x,
        -derivative * point.y, 1.0));
}

bool lensIntersect(Lens* lens, uint surfaceIndex, float3 origin,
    float3 direction, out float3 hit)
{
    hit = float3(0.0);
    const Surface surface = lens->surfaces[surfaceIndex];
    float t = (surface.z - origin.z) / direction.z;
    if (!(t > 0.0))
        return false;
    [loop]
    for (uint iteration = 0u; iteration < 12u; ++iteration)
    {
        const float3 point = origin + t * direction;
        const float f = point.z - surface.z
            - lensSag(lens, surfaceIndex, dot(point.xy, point.xy));
        const float3 normal = lensSurfaceNormal(lens, surfaceIndex, point);
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
    return dot(hit.xy, hit.xy) <= surface.apertureRadius
        * surface.apertureRadius;
}

float lensMediumIor(Lens* lens, uint mediumIndex, float wavelengthNm)
{
    const Medium medium = lens->media[mediumIndex];
    return sellmeierIor(medium.b, medium.c, wavelengthNm);
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

bool traceLensFilmToWorld(Lens* lens, inout float3 origin,
    inout float3 direction, float wavelengthNm)
{
    const uint surfaceCount = min(lens->surfaceCount, MaxLensSurfaces);
    const uint mediumCount = max(min(lens->mediumCount, MaxLensSurfaces + 1u), 1u);
    uint medium = 0u;
    [loop]
    for (uint surfaceIndex = 0u; surfaceIndex < surfaceCount; ++surfaceIndex)
    {
        float3 hit;
        if (!lensIntersect(lens, surfaceIndex, origin, direction, hit))
            return false;
        const uint nextMedium = min(lens->surfaces[surfaceIndex].mediumAfter,
            mediumCount - 1u);
        float3 nextDirection;
        if (!lensRefract(direction, lensSurfaceNormal(lens, surfaceIndex, hit),
                lensMediumIor(lens, medium, wavelengthNm),
                lensMediumIor(lens, nextMedium, wavelengthNm), nextDirection))
            return false;
        origin = hit;
        direction = nextDirection;
        medium = nextMedium;
    }
    return surfaceCount != 0u;
}
