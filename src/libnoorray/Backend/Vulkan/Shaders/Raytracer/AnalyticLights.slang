#pragma once

#include "RgbToSpectrum.slang"
#include "Sampling.slang"

static const float AnalyticLightPi = 3.14159265358979323846;

struct AnalyticLightSample
{
    float3 direction;
    float distance;
    SampledSpectrum radianceOverSelectionPdf;
    uint instanceIndex;
    uint primitiveIndex;
    float2 barycentrics;
    float emissionMultiplier;
    bool mesh;
    bool valid;
};

struct AnalyticLightHit
{
    SampledSpectrum radiance;
    float distance;
    float pdf;
    bool valid;
};

float lightFloat(ByteAddressBuffer lights, uint offset)
{
    return asfloat(lights.Load(offset));
}

float3 lightFloat3(ByteAddressBuffer lights, uint offset)
{
    return float3(lightFloat(lights, offset), lightFloat(lights, offset + 4u),
        lightFloat(lights, offset + 8u));
}

float3 sampleConeDirection(float3 axis, float oneMinusCosine, float2 u)
{
    const float cosTheta = 1.0 - u.x * oneMinusCosine;
    const float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    const float phi = 2.0 * AnalyticLightPi * u.y;
    const float3 n = normalize(axis);
    const float3 helper = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0)
                                           : float3(0.0, 1.0, 0.0);
    const float3 tangent = normalize(cross(helper, n));
    const float3 bitangent = cross(n, tangent);
    return normalize(n * cosTheta + tangent * (cos(phi) * sinTheta)
        + bitangent * (sin(phi) * sinTheta));
}

float oneMinusCosineFromSineSquared(float sineSquared)
{
    return sineSquared < 0.5
        ? sineSquared / (1.0 + sqrt(max(1.0 - sineSquared, 0.0)))
        : 1.0 - sqrt(max(1.0 - sineSquared, 0.0));
}

bool sampleSphereLight(float3 origin, float3 center, float radius, float2 u,
    out float3 direction, out float distance, out float pdf)
{
    direction = float3(0.0);
    distance = 0.0;
    pdf = 0.0;
    const float3 toCenter = center - origin;
    const float centerDistanceSquared = dot(toCenter, toCenter);
    if (radius <= 0.0)
    {
        distance = sqrt(centerDistanceSquared);
        direction = distance > 0.0 ? toCenter / distance : float3(0.0);
        pdf = 0.0;
        return distance > 0.0;
    }
    if (centerDistanceSquared > radius * radius)
    {
        const float centerDistance = sqrt(centerDistanceSquared);
        const float sineSquared = min(radius * radius / centerDistanceSquared, 1.0);
        const float oneMinusCosine = oneMinusCosineFromSineSquared(sineSquared);
        direction = sampleConeDirection(toCenter / centerDistance, oneMinusCosine, u);
        const float cosine = dot(direction, toCenter / centerDistance);
        distance = centerDistance * cosine - sqrt(max(radius * radius
            - centerDistanceSquared + centerDistanceSquared * cosine * cosine, 0.0));
        pdf = 1.0 / max(2.0 * AnalyticLightPi * oneMinusCosine, 1.0e-20);
        return distance > 0.0;
    }
    const float z = 1.0 - 2.0 * u.x;
    const float radial = sqrt(max(1.0 - z * z, 0.0));
    const float phi = 2.0 * AnalyticLightPi * u.y;
    direction = float3(radial * cos(phi), radial * sin(phi), z);
    const float3 offset = origin - center;
    const float halfB = dot(offset, direction);
    const float discriminant = halfB * halfB - (dot(offset, offset) - radius * radius);
    if (discriminant < 0.0)
        return false;
    distance = -halfB + sqrt(discriminant);
    pdf = 1.0 / (4.0 * AnalyticLightPi);
    return distance > 0.0;
}

AnalyticLightSample sampleAnalyticLight(ByteAddressBuffer lights,
    ByteAddressBuffer spectralTables, float3 position,
    SampledWavelengths wavelengths, inout Rng rng)
{
    AnalyticLightSample result;
    result.direction = float3(0.0);
    result.distance = 0.0;
    result.radianceOverSelectionPdf = SampledSpectrum(0.0);
    result.instanceIndex = 0u;
    result.primitiveIndex = 0u;
    result.barycentrics = float2(0.0);
    result.emissionMultiplier = 0.0;
    result.mesh = false;
    result.valid = false;

    const uint count = lights.Load(0u);
    if (count == 0u)
        return result;
    const float finiteWeight = max(lightFloat(lights, 4u), 0.0);
    if (finiteWeight <= 0.0)
        return result;
    const float target = rng.nextFloat() * finiteWeight;
    uint selected = count - 1u;
    float accumulatedWeight = 0.0;
    [loop]
    for (uint i = 0u; i < count; ++i)
    {
        accumulatedWeight += max(lightFloat(lights, 16u + i * 96u + 92u), 0.0);
        if (target <= accumulatedWeight)
        {
            selected = i;
            break;
        }
    }
    const uint base = 16u + selected * 96u;
    const uint type = lights.Load(base + 12u);
    const float3 lightPosition = lightFloat3(lights, base);
    const float3 authoredDirection = lightFloat3(lights, base + 16u);
    const uint flags = lights.Load(base + 28u);
    const float3 color = max(lightFloat3(lights, base + 32u), 0.0);
    const float intensity = max(lightFloat(lights, base + 44u), 0.0);
    float3 estimatorRgb = float3(0.0);

    if (type == 0u || type == 1u)
    {
        const float radius = max(lightFloat(lights, base + 64u), 0.0);
        float lightPdf = 0.0;
        if (!sampleSphereLight(position, lightPosition, radius, rng.nextFloat2(),
                result.direction, result.distance, lightPdf))
            return result;
        float attenuation = 1.0;
        if (type == 1u)
        {
            const float inner = lightFloat(lights, base + 68u)
                * AnalyticLightPi / 180.0;
            const float outer = lightFloat(lights, base + 72u)
                * AnalyticLightPi / 180.0;
            const float cosine = dot(normalize(authoredDirection), -result.direction);
            const float innerCos = cos(inner);
            const float outerCos = cos(outer);
            const float cone = saturate((cosine - outerCos)
                / max(innerCos - outerCos, 1.0e-6));
            attenuation = cone * cone * (3.0 - 2.0 * cone);
        }
        estimatorRgb = radius > 0.0
            ? color * (intensity * attenuation
                / max(AnalyticLightPi * radius * radius * lightPdf, 1.0e-20))
            : color * (intensity * attenuation
                / max(result.distance * result.distance, 1.0e-12));
    }
    else if (type == 2u)
    {
        const float width = max(lightFloat(lights, base + 64u), 0.0);
        const float height = max(lightFloat(lights, base + 68u), 0.0);
        const float area = width * height;
        if (area <= 0.0)
            return result;
        const float3 normal = normalize(authoredDirection);
        const float3 tangent = normalize(lightFloat3(lights, base + 48u));
        const float3 bitangent = normalize(cross(normal, tangent));
        const float2 square = rng.nextFloat2() - 0.5;
        const float3 sampledPosition = lightPosition
            + tangent * (square.x * width) + bitangent * (square.y * height);
        const float3 delta = sampledPosition - position;
        const float distanceSquared = dot(delta, delta);
        if (distanceSquared <= 1.0e-12)
            return result;
        result.distance = sqrt(distanceSquared);
        result.direction = delta / result.distance;
        float emitterCosine = dot(normal, -result.direction);
        emitterCosine = (flags & 1u) != 0u ? abs(emitterCosine)
                                           : max(emitterCosine, 0.0);
        if (emitterCosine <= 0.0)
            return result;
        float barnDoorMask = 1.0;
        const float barnDoorAngle = lightFloat(lights, base + 76u);
        const float barnDoorLength = max(lightFloat(lights, base + 80u), 0.0);
        if (barnDoorLength > 0.0 && barnDoorAngle < 89.9)
        {
            const float3 outgoing = -result.direction;
            const float forward = max(abs(dot(normal, outgoing)), 1.0e-5);
            const float3 local = sampledPosition - lightPosition;
            const float expansion = barnDoorLength * tan(max(barnDoorAngle, 0.0)
                * AnalyticLightPi / 180.0);
            const float projectedU = dot(local, tangent) + barnDoorLength
                * dot(outgoing, tangent) / forward;
            const float projectedV = dot(local, bitangent) + barnDoorLength
                * dot(outgoing, bitangent) / forward;
            if (abs(projectedU) > width * 0.5 + expansion
                || abs(projectedV) > height * 0.5 + expansion)
                barnDoorMask = 0.0;
        }
        estimatorRgb = color * (intensity * barnDoorMask * area
            * emitterCosine / distanceSquared);
    }
    else if (type == 3u)
    {
        const float3 centerDirection = -normalize(authoredDirection);
        const float diameter = clamp(lightFloat(lights, base + 64u), 0.0, 180.0);
        result.direction = centerDirection;
        result.distance = 1.0e16;
        estimatorRgb = color * intensity;
        if (diameter > 0.0)
        {
            const float halfAngle = 0.5 * diameter * AnalyticLightPi / 180.0;
            const float oneMinusCosine = 1.0 - cos(halfAngle);
            result.direction = sampleConeDirection(centerDirection,
                oneMinusCosine, rng.nextFloat2());
            const float pdf = 1.0 / max(2.0 * AnalyticLightPi * oneMinusCosine,
                1.0e-20);
            const float projectedArea = AnalyticLightPi
                * sin(halfAngle) * sin(halfAngle);
            estimatorRgb /= max(projectedArea * pdf, 1.0e-20);
        }
    }
    else if (type == 4u)
    {
        const float3 a = lightPosition;
        const float3 b = authoredDirection;
        const float3 c = lightFloat3(lights, base + 32u);
        const float2 u = rng.nextFloat2();
        const float rootU = sqrt(u.x);
        const float2 barycentrics = float2(rootU * (1.0 - u.y), rootU * u.y);
        const float3 sampledPosition = a * (1.0 - barycentrics.x - barycentrics.y)
            + b * barycentrics.x + c * barycentrics.y;
        const float3 delta = sampledPosition - position;
        const float distanceSquared = dot(delta, delta);
        const float area = max(intensity, 0.0);
        if (distanceSquared <= 1.0e-12 || area <= 0.0)
            return result;
        result.distance = sqrt(distanceSquared);
        result.direction = delta / result.distance;
        const float3 normal = normalize(cross(b - a, c - a));
        const float emitterCosine = max(dot(normal, -result.direction), 0.0);
        if (emitterCosine <= 0.0)
            return result;
        result.instanceIndex = flags;
        result.primitiveIndex = lights.Load(base + 60u);
        result.barycentrics = barycentrics;
        result.emissionMultiplier = area * emitterCosine / distanceSquared;
        result.mesh = true;
    }
    else
        return result;

    const float selectedWeight = max(lightFloat(lights, base + 92u), 0.0);
    const float selectionReciprocal = finiteWeight / max(selectedWeight, 1.0e-20);
    estimatorRgb *= selectionReciprocal;
    result.emissionMultiplier *= selectionReciprocal;
    result.radianceOverSelectionPdf = rgbIlluminantToSpectrum(
        spectralTables, estimatorRgb, wavelengths);
    result.valid = result.mesh
        ? result.emissionMultiplier > 0.0
        : spectrumMaxComponent(result.radianceOverSelectionPdf) > 0.0;
    return result;
}

float meshLightHitPdf(ByteAddressBuffer lights, uint instanceIndex,
    uint primitiveIndex, float3 origin, float3 hitPosition)
{
    const uint count = lights.Load(0u);
    const float finiteWeight = max(lightFloat(lights, 4u), 0.0);
    if (finiteWeight <= 0.0)
        return 0.0;
    [loop]
    for (uint i = 0u; i < count; ++i)
    {
        const uint base = 16u + i * 96u;
        if (lights.Load(base + 12u) != 4u
            || lights.Load(base + 28u) != instanceIndex
            || lights.Load(base + 60u) != primitiveIndex)
            continue;
        const float3 a = lightFloat3(lights, base);
        const float3 b = lightFloat3(lights, base + 16u);
        const float3 c = lightFloat3(lights, base + 32u);
        const float area = max(lightFloat(lights, base + 44u), 0.0);
        const float3 delta = hitPosition - origin;
        const float distanceSquared = dot(delta, delta);
        if (area <= 0.0 || distanceSquared <= 1.0e-12)
            return 0.0;
        const float cosine = max(dot(normalize(cross(b - a, c - a)),
            -delta / sqrt(distanceSquared)), 0.0);
        const float conditionalPdf = cosine > 0.0
            ? distanceSquared / max(area * cosine, 1.0e-20) : 0.0;
        const float selectionPdf = max(lightFloat(lights, base + 92u), 0.0)
            / finiteWeight;
        return selectionPdf * conditionalPdf;
    }
    return 0.0;
}

AnalyticLightHit intersectAnalyticLights(ByteAddressBuffer lights,
    ByteAddressBuffer spectralTables, float3 origin, float3 direction,
    float minimumDistance, float maximumDistance, SampledWavelengths wavelengths)
{
    AnalyticLightHit result;
    result.radiance = SampledSpectrum(0.0);
    result.distance = maximumDistance;
    result.pdf = 0.0;
    result.valid = false;
    const uint count = lights.Load(0u);
    const float finiteWeight = max(lightFloat(lights, 4u), 0.0);
    [loop]
    for (uint i = 0u; i < count; ++i)
    {
        const uint base = 16u + i * 96u;
        const uint type = lights.Load(base + 12u);
        if (type == 4u)
            continue;
        const float3 center = lightFloat3(lights, base);
        const float3 authoredDirection = lightFloat3(lights, base + 16u);
        const uint flags = lights.Load(base + 28u);
        const float3 color = max(lightFloat3(lights, base + 32u), 0.0);
        const float intensity = max(lightFloat(lights, base + 44u), 0.0);
        float distance = maximumDistance;
        float conditionalPdf = 0.0;
        float3 radianceRgb = float3(0.0);
        bool hit = false;
        if (type == 0u || type == 1u)
        {
            const float radius = max(lightFloat(lights, base + 64u), 0.0);
            if (radius <= 0.0)
                continue;
            const float3 offset = origin - center;
            const float halfB = dot(offset, direction);
            const float c = dot(offset, offset) - radius * radius;
            const float discriminant = halfB * halfB - c;
            if (discriminant < 0.0)
                continue;
            const float root = sqrt(discriminant);
            const float nearDistance = -halfB - root;
            const float farDistance = -halfB + root;
            distance = nearDistance >= minimumDistance ? nearDistance : farDistance;
            if (distance < minimumDistance || distance > result.distance)
                continue;
            const float centerDistanceSquared = dot(center - origin, center - origin);
            conditionalPdf = centerDistanceSquared <= radius * radius
                ? 1.0 / (4.0 * AnalyticLightPi)
                : 1.0 / max(2.0 * AnalyticLightPi
                    * oneMinusCosineFromSineSquared(min(radius * radius
                        / centerDistanceSquared, 1.0)), 1.0e-20);
            float attenuation = 1.0;
            if (type == 1u)
            {
                const float inner = lightFloat(lights, base + 68u)
                    * AnalyticLightPi / 180.0;
                const float outer = lightFloat(lights, base + 72u)
                    * AnalyticLightPi / 180.0;
                const float cosine = dot(normalize(authoredDirection),
                    normalize(origin - (origin + direction * distance)));
                const float cone = saturate((cosine - cos(outer))
                    / max(cos(inner) - cos(outer), 1.0e-6));
                attenuation = cone * cone * (3.0 - 2.0 * cone);
            }
            radianceRgb = color * intensity * attenuation
                / max(AnalyticLightPi * radius * radius, 1.0e-20);
            hit = true;
        }
        else if (type == 2u)
        {
            const float width = max(lightFloat(lights, base + 64u), 0.0);
            const float height = max(lightFloat(lights, base + 68u), 0.0);
            const float area = width * height;
            const float3 normal = normalize(authoredDirection);
            const float denominator = dot(normal, direction);
            if (area <= 0.0 || abs(denominator) <= 1.0e-8)
                continue;
            distance = dot(center - origin, normal) / denominator;
            if (distance < minimumDistance || distance > result.distance)
                continue;
            const float3 tangent = normalize(lightFloat3(lights, base + 48u));
            const float3 bitangent = normalize(cross(normal, tangent));
            const float3 local = origin + direction * distance - center;
            if (abs(dot(local, tangent)) > 0.5 * width
                || abs(dot(local, bitangent)) > 0.5 * height)
                continue;
            const float cosine = (flags & 1u) != 0u
                ? abs(dot(normal, -direction)) : max(dot(normal, -direction), 0.0);
            if (cosine <= 0.0)
                continue;
            conditionalPdf = distance * distance / max(area * cosine, 1.0e-20);
            radianceRgb = color * intensity;
            hit = true;
        }
        else if (type == 3u && result.distance >= maximumDistance)
        {
            const float3 axis = -normalize(authoredDirection);
            const float halfAngle = 0.5 * clamp(lightFloat(lights, base + 64u),
                0.0, 180.0) * AnalyticLightPi / 180.0;
            const float oneMinusCosine = 1.0 - cos(halfAngle);
            const float cosine = dot(direction, axis);
            if ((oneMinusCosine <= 0.0 && cosine < 1.0 - 1.0e-7)
                || (oneMinusCosine > 0.0 && 1.0 - cosine > oneMinusCosine))
                continue;
            distance = maximumDistance;
            if (oneMinusCosine > 0.0)
            {
                conditionalPdf = 1.0 / max(2.0 * AnalyticLightPi
                    * oneMinusCosine, 1.0e-20);
                const float projectedArea = AnalyticLightPi
                    * sin(halfAngle) * sin(halfAngle);
                radianceRgb = color * intensity / max(projectedArea, 1.0e-20);
            }
            else
                radianceRgb = color * intensity;
            hit = true;
        }
        if (!hit)
            continue;
        const float selectionPdf = finiteWeight > 0.0
            ? max(lightFloat(lights, base + 92u), 0.0) / finiteWeight : 0.0;
        result.radiance = rgbIlluminantToSpectrum(spectralTables,
            radianceRgb, wavelengths);
        result.distance = distance;
        result.pdf = selectionPdf * conditionalPdf;
        result.valid = spectrumMaxComponent(result.radiance) > 0.0;
    }
    return result;
}
