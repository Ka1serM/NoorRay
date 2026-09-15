#pragma once

#include "ByteBuffer.slang"


#include "../../Shared/Light.h"

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

// The scene's analytic lights remain separated by kind all the way to the
// shader. This keeps light-specific fields readable and avoids interpreting a
// type-dependent parameter array at every sample or intersection.
struct LightTable
{
    PointLight* points;
    uint pointCount;
    SpotLight* spots;
    uint spotCount;
    RectLight* rects;
    uint rectCount;
    DirectionalLight* directionals;
    uint directionalCount;
    MeshLight* meshes;
    uint meshCount;
    float finiteWeight;
};

LightTable makeLightTable(Frame frame)
{
    return {frame.pointLights, frame.pointLightCount,
        frame.spotLights, frame.spotLightCount,
        frame.rectLights, frame.rectLightCount,
        frame.directionalLights, frame.directionalLightCount,
        frame.meshLights, frame.meshLightCount,
        frame.lightFiniteWeight};
}

bool hasAnalyticLights(Frame frame)
{
    return frame.pointLightCount != 0u || frame.spotLightCount != 0u
        || frame.rectLightCount != 0u || frame.directionalLightCount != 0u
        || frame.meshLightCount != 0u;
}

struct LightChoice
{
    uint kind;
    uint index;
    float weight;
};

LightChoice chooseLight(LightTable lights, float target)
{
    LightChoice choice = {0u, 0u, 0.0};
    float accumulated = 0.0;
    [loop]
    for (uint i = 0u; i < lights.pointCount; ++i)
    {
        const float weight = max(lights.points[i].selectionWeight, 0.0);
        accumulated += weight;
        choice = {0u, i, weight};
        if (target <= accumulated)
            return choice;
    }
    [loop]
    for (uint i = 0u; i < lights.spotCount; ++i)
    {
        const float weight = max(lights.spots[i].selectionWeight, 0.0);
        accumulated += weight;
        choice = {1u, i, weight};
        if (target <= accumulated)
            return choice;
    }
    [loop]
    for (uint i = 0u; i < lights.rectCount; ++i)
    {
        const float weight = max(lights.rects[i].selectionWeight, 0.0);
        accumulated += weight;
        choice = {2u, i, weight};
        if (target <= accumulated)
            return choice;
    }
    [loop]
    for (uint i = 0u; i < lights.directionalCount; ++i)
    {
        const float weight = max(lights.directionals[i].selectionWeight, 0.0);
        accumulated += weight;
        choice = {3u, i, weight};
        if (target <= accumulated)
            return choice;
    }
    [loop]
    for (uint i = 0u; i < lights.meshCount; ++i)
    {
        const float weight = max(lights.meshes[i].selectionWeight, 0.0);
        accumulated += weight;
        choice = {4u, i, weight};
        if (target <= accumulated)
            return choice;
    }
    return choice;
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

AnalyticLightSample sampleAnalyticLight(LightTable lights,
    ByteBuffer spectralTables, float3 position,
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

    if (lights.pointCount == 0u && lights.spotCount == 0u
        && lights.rectCount == 0u && lights.directionalCount == 0u
        && lights.meshCount == 0u)
        return result;
    const float finiteWeight = max(lights.finiteWeight, 0.0);
    if (finiteWeight <= 0.0)
        return result;
    const LightChoice choice = chooseLight(lights, rng.nextFloat() * finiteWeight);
    const float selectedWeight = max(choice.weight, 0.0);
    if (selectedWeight <= 0.0)
        return result;
    float3 estimatorRgb = float3(0.0);

    if (choice.kind == 0u)
    {
        const PointLight light = lights.points[choice.index];
        const float radius = max(light.softRadius, 0.0);
        float lightPdf = 0.0;
        if (!sampleSphereLight(position, light.position, radius, rng.nextFloat2(),
                result.direction, result.distance, lightPdf))
            return result;
        const float3 lightColor = max(light.color, 0.0);
        estimatorRgb = radius > 0.0
            ? lightColor * (max(light.intensity, 0.0)
                / max(AnalyticLightPi * radius * radius * lightPdf, 1.0e-20))
            : lightColor * max(light.intensity, 0.0)
                / max(result.distance * result.distance, 1.0e-12);
    }
    else if (choice.kind == 1u)
    {
        const SpotLight light = lights.spots[choice.index];
        const float radius = max(light.softRadius, 0.0);
        float lightPdf = 0.0;
        if (!sampleSphereLight(position, light.position, radius, rng.nextFloat2(),
                result.direction, result.distance, lightPdf))
            return result;
        const float inner = light.innerConeAngle * AnalyticLightPi / 180.0;
        const float outer = light.outerConeAngle * AnalyticLightPi / 180.0;
        const float cosine = dot(normalize(light.direction), -result.direction);
        const float cone = saturate((cosine - cos(outer))
            / max(cos(inner) - cos(outer), 1.0e-6));
        const float attenuation = cone * cone * (3.0 - 2.0 * cone);
        const float3 lightColor = max(light.color, 0.0);
        estimatorRgb = radius > 0.0
            ? lightColor * (max(light.intensity, 0.0) * attenuation
                / max(AnalyticLightPi * radius * radius * lightPdf, 1.0e-20))
            : lightColor * (max(light.intensity, 0.0) * attenuation
                / max(result.distance * result.distance, 1.0e-12));
    }
    else if (choice.kind == 2u)
    {
        const RectLight light = lights.rects[choice.index];
        const float width = max(light.width, 0.0);
        const float height = max(light.height, 0.0);
        const float area = width * height;
        if (area <= 0.0)
            return result;
        const float3 normal = normalize(light.direction);
        const float3 tangent = normalize(light.tangent);
        const float3 bitangent = normalize(cross(normal, tangent));
        const float2 square = rng.nextFloat2() - 0.5;
        const float3 sampledPosition = light.position
            + tangent * (square.x * width) + bitangent * (square.y * height);
        const float3 delta = sampledPosition - position;
        const float distanceSquared = dot(delta, delta);
        if (distanceSquared <= 1.0e-12)
            return result;
        result.distance = sqrt(distanceSquared);
        result.direction = delta / result.distance;
        float emitterCosine = dot(normal, -result.direction);
        emitterCosine = light.twoSided != 0u ? abs(emitterCosine)
                                           : max(emitterCosine, 0.0);
        if (emitterCosine <= 0.0)
            return result;
        float barnDoorMask = 1.0;
        const float barnDoorAngle = light.barnDoorAngle;
        const float barnDoorLength = max(light.barnDoorLength, 0.0);
        if (barnDoorLength > 0.0 && barnDoorAngle < 89.9)
        {
            const float3 outgoing = -result.direction;
            const float forward = max(abs(dot(normal, outgoing)), 1.0e-5);
            const float3 local = sampledPosition - light.position;
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
        estimatorRgb = max(light.color, 0.0) * (max(light.intensity, 0.0)
            * barnDoorMask * area
            * emitterCosine / distanceSquared);
    }
    else if (choice.kind == 3u)
    {
        const DirectionalLight light = lights.directionals[choice.index];
        const float3 centerDirection = -normalize(light.direction);
        const float diameter = clamp(light.softAngle, 0.0, 180.0);
        result.direction = centerDirection;
        result.distance = 1.0e16;
        estimatorRgb = max(light.color, 0.0) * max(light.intensity, 0.0);
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
    else if (choice.kind == 4u)
    {
        const MeshLight light = lights.meshes[choice.index];
        const float3 a = light.a;
        const float3 b = light.b;
        const float3 c = light.c;
        const float2 u = rng.nextFloat2();
        const float rootU = sqrt(u.x);
        const float2 barycentrics = float2(rootU * (1.0 - u.y), rootU * u.y);
        const float3 sampledPosition = a * (1.0 - barycentrics.x - barycentrics.y)
            + b * barycentrics.x + c * barycentrics.y;
        const float3 delta = sampledPosition - position;
        const float distanceSquared = dot(delta, delta);
        const float area = max(light.area, 0.0);
        if (distanceSquared <= 1.0e-12 || area <= 0.0)
            return result;
        result.distance = sqrt(distanceSquared);
        result.direction = delta / result.distance;
        const float3 normal = normalize(cross(b - a, c - a));
        const float emitterCosine = max(dot(normal, -result.direction), 0.0);
        if (emitterCosine <= 0.0)
            return result;
        result.instanceIndex = light.instanceIndex;
        result.primitiveIndex = light.primitiveIndex;
        result.barycentrics = barycentrics;
        result.emissionMultiplier = area * emitterCosine / distanceSquared;
        result.mesh = true;
    }
    else
        return result;

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

float meshLightHitPdf(LightTable lights, uint instanceIndex,
    uint primitiveIndex, float3 origin, float3 hitPosition)
{
    const float finiteWeight = max(lights.finiteWeight, 0.0);
    if (finiteWeight <= 0.0)
        return 0.0;
    [loop]
    for (uint i = 0u; i < lights.meshCount; ++i)
    {
        const MeshLight light = lights.meshes[i];
        if (light.instanceIndex != instanceIndex
            || light.primitiveIndex != primitiveIndex)
            continue;
        const float3 a = light.a;
        const float3 b = light.b;
        const float3 c = light.c;
        const float area = max(light.area, 0.0);
        const float3 delta = hitPosition - origin;
        const float distanceSquared = dot(delta, delta);
        if (area <= 0.0 || distanceSquared <= 1.0e-12)
            return 0.0;
        const float cosine = max(dot(normalize(cross(b - a, c - a)),
            -delta / sqrt(distanceSquared)), 0.0);
        const float conditionalPdf = cosine > 0.0
            ? distanceSquared / max(area * cosine, 1.0e-20) : 0.0;
        const float selectionPdf = max(light.selectionWeight, 0.0)
            / finiteWeight;
        return selectionPdf * conditionalPdf;
    }
    return 0.0;
}

bool intersectSphereLight(float3 center, float radius, float3 origin,
    float3 direction, float minimumDistance, float maximumDistance,
    out float distance, out float conditionalPdf)
{
    distance = maximumDistance;
    conditionalPdf = 0.0;
    if (radius <= 0.0)
        return false;
    const float3 offset = origin - center;
    const float halfB = dot(offset, direction);
    const float discriminant = halfB * halfB
        - (dot(offset, offset) - radius * radius);
    if (discriminant < 0.0)
        return false;
    const float root = sqrt(discriminant);
    const float nearDistance = -halfB - root;
    const float farDistance = -halfB + root;
    distance = nearDistance >= minimumDistance ? nearDistance : farDistance;
    if (distance < minimumDistance || distance > maximumDistance)
        return false;
    const float centerDistanceSquared = dot(center - origin, center - origin);
    conditionalPdf = centerDistanceSquared <= radius * radius
        ? 1.0 / (4.0 * AnalyticLightPi)
        : 1.0 / max(2.0 * AnalyticLightPi
            * oneMinusCosineFromSineSquared(min(radius * radius
                / centerDistanceSquared, 1.0)), 1.0e-20);
    return true;
}

void publishAnalyticLightHit(inout AnalyticLightHit result,
    ByteBuffer spectralTables, float3 radianceRgb, float distance,
    float conditionalPdf, float selectionWeight,
    SampledWavelengths wavelengths, float finiteWeight)
{
    result.radiance = rgbIlluminantToSpectrum(spectralTables,
        radianceRgb, wavelengths);
    result.distance = distance;
    result.pdf = (finiteWeight > 0.0
        ? max(selectionWeight, 0.0) / finiteWeight : 0.0) * conditionalPdf;
    result.valid = spectrumMaxComponent(result.radiance) > 0.0;
}

AnalyticLightHit intersectAnalyticLights(LightTable lights,
    ByteBuffer spectralTables, float3 origin, float3 direction,
    float minimumDistance, float maximumDistance, SampledWavelengths wavelengths)
{
    AnalyticLightHit result;
    result.radiance = SampledSpectrum(0.0);
    result.distance = maximumDistance;
    result.pdf = 0.0;
    result.valid = false;
    const float finiteWeight = max(lights.finiteWeight, 0.0);

    [loop]
    for (uint i = 0u; i < lights.pointCount; ++i)
    {
        const PointLight light = lights.points[i];
        float distance;
        float conditionalPdf;
        if (!intersectSphereLight(light.position, light.softRadius, origin,
                direction, minimumDistance, result.distance, distance,
                conditionalPdf))
            continue;
        publishAnalyticLightHit(result, spectralTables,
            max(light.color, 0.0) * max(light.intensity, 0.0)
                / max(AnalyticLightPi * light.softRadius * light.softRadius, 1.0e-20),
            distance, conditionalPdf, light.selectionWeight, wavelengths,
            finiteWeight);
    }

    [loop]
    for (uint i = 0u; i < lights.spotCount; ++i)
    {
        const SpotLight light = lights.spots[i];
        float distance;
        float conditionalPdf;
        if (!intersectSphereLight(light.position, light.softRadius, origin,
                direction, minimumDistance, result.distance, distance,
                conditionalPdf))
            continue;
        const float inner = light.innerConeAngle * AnalyticLightPi / 180.0;
        const float outer = light.outerConeAngle * AnalyticLightPi / 180.0;
        const float cosine = dot(normalize(light.direction),
            normalize(origin - (origin + direction * distance)));
        const float cone = saturate((cosine - cos(outer))
            / max(cos(inner) - cos(outer), 1.0e-6));
        const float attenuation = cone * cone * (3.0 - 2.0 * cone);
        publishAnalyticLightHit(result, spectralTables,
            max(light.color, 0.0) * max(light.intensity, 0.0) * attenuation
                / max(AnalyticLightPi * light.softRadius * light.softRadius, 1.0e-20),
            distance, conditionalPdf, light.selectionWeight, wavelengths,
            finiteWeight);
    }

    [loop]
    for (uint i = 0u; i < lights.rectCount; ++i)
    {
        const RectLight light = lights.rects[i];
        const float width = max(light.width, 0.0);
        const float height = max(light.height, 0.0);
        const float area = width * height;
        const float3 normal = normalize(light.direction);
        const float denominator = dot(normal, direction);
        if (area <= 0.0 || abs(denominator) <= 1.0e-8)
            continue;
        const float distance = dot(light.position - origin, normal) / denominator;
        if (distance < minimumDistance || distance > result.distance)
            continue;
        const float3 tangent = normalize(light.tangent);
        const float3 bitangent = normalize(cross(normal, tangent));
        const float3 local = origin + direction * distance - light.position;
        if (abs(dot(local, tangent)) > 0.5 * width
            || abs(dot(local, bitangent)) > 0.5 * height)
            continue;
        const float cosine = light.twoSided != 0u
            ? abs(dot(normal, -direction)) : max(dot(normal, -direction), 0.0);
        if (cosine <= 0.0)
            continue;
        publishAnalyticLightHit(result, spectralTables,
            max(light.color, 0.0) * max(light.intensity, 0.0), distance,
            distance * distance / max(area * cosine, 1.0e-20),
            light.selectionWeight, wavelengths, finiteWeight);
    }

    [loop]
    for (uint i = 0u; i < lights.directionalCount; ++i)
    {
        const DirectionalLight light = lights.directionals[i];
        if (result.distance < maximumDistance)
            continue;
        const float3 axis = -normalize(light.direction);
        const float halfAngle = 0.5 * clamp(light.softAngle, 0.0, 180.0)
            * AnalyticLightPi / 180.0;
        const float oneMinusCosine = 1.0 - cos(halfAngle);
        const float cosine = dot(direction, axis);
        if ((oneMinusCosine <= 0.0 && cosine < 1.0 - 1.0e-7)
            || (oneMinusCosine > 0.0 && 1.0 - cosine > oneMinusCosine))
            continue;
        float conditionalPdf = 0.0;
        float3 radianceRgb = max(light.color, 0.0) * max(light.intensity, 0.0);
        if (oneMinusCosine > 0.0)
        {
            conditionalPdf = 1.0 / max(2.0 * AnalyticLightPi
                * oneMinusCosine, 1.0e-20);
            const float projectedArea = AnalyticLightPi
                * sin(halfAngle) * sin(halfAngle);
            radianceRgb /= max(projectedArea, 1.0e-20);
        }
        publishAnalyticLightHit(result, spectralTables, radianceRgb,
            maximumDistance, conditionalPdf, light.selectionWeight,
            wavelengths, finiteWeight);
    }
    return result;
}
