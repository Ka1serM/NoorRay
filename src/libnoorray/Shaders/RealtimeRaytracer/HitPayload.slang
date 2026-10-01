#pragma once

#include "../Raytracer/RaytracingAbi.slang"
#include "../Raytracer/RayDifferential.slang"

// Light passing fractional surfaces below this fraction counts as blocked.
static const float RealtimeMinTransmittance = 1.0e-3;

// Encoding of RealtimeHitPayload. The payload is live in registers across
// every realtime TraceRay, so it carries the material's closure already
// resolved to one normal and one roughness, with unit vectors as 16-bit
// octahedral pairs and reflectances as halves.

uint packHalf2(float x, float y)
{
    return f32tof16(x) | (f32tof16(y) << 16);
}

float2 unpackHalf2(uint packed)
{
    return float2(f16tof32(packed & 0xFFFFu), f16tof32(packed >> 16));
}

float2 octahedralSign(float2 value)
{
    return float2(value.x >= 0.0 ? 1.0 : -1.0, value.y >= 0.0 ? 1.0 : -1.0);
}

uint packUnitVector(float3 direction)
{
    float2 p = direction.xy / (abs(direction.x) + abs(direction.y) + abs(direction.z));
    if (direction.z < 0.0)
        p = (1.0 - abs(p.yx)) * octahedralSign(p);
    const int2 q = int2(round(clamp(p, -1.0, 1.0) * 32767.0));
    return (uint(q.x) & 0xFFFFu) | (uint(q.y) << 16);
}

float3 unpackUnitVector(uint packed)
{
    const float2 p = float2(float(int(packed << 16) >> 16), float(int(packed) >> 16)) / 32767.0;
    float3 direction = float3(p, 1.0 - abs(p.x) - abs(p.y));
    if (direction.z < 0.0)
        direction.xy = (1.0 - abs(direction.yx)) * octahedralSign(direction.xy);
    return normalize(direction);
}

// A closure's Lambertian diffuse and GGX reflection lobes. RealtimeHitPayload
// and RealtimeSurface carry it packed in five words of halves.
struct RealtimeClosure
{
    float3 diffuse;
    float roughness;
    float3 specularF0;
    float3 specularF90;
};

void packRealtimeClosure(out uint words[5], RealtimeClosure closure)
{
    words[0] = packHalf2(closure.diffuse.r, closure.diffuse.g);
    words[1] = packHalf2(closure.diffuse.b, closure.roughness);
    words[2] = packHalf2(closure.specularF0.r, closure.specularF0.g);
    words[3] = packHalf2(closure.specularF0.b, closure.specularF90.r);
    words[4] = packHalf2(closure.specularF90.g, closure.specularF90.b);
}

RealtimeClosure unpackRealtimeClosure(uint words[5])
{
    const float2 diffuseRG = unpackHalf2(words[0]);
    const float2 diffuseBRoughness = unpackHalf2(words[1]);
    const float2 f0RG = unpackHalf2(words[2]);
    const float2 f0BF90R = unpackHalf2(words[3]);
    const float2 f90GB = unpackHalf2(words[4]);
    RealtimeClosure closure;
    closure.diffuse = float3(diffuseRG, diffuseBRoughness.x);
    closure.roughness = diffuseBRoughness.y;
    closure.specularF0 = float3(f0RG, f0BF90R.x);
    closure.specularF90 = float3(f0BF90R.y, f90GB);
    return closure;
}

// Differentials as halves. Grazing hits can exceed their range, which is
// clamped so that no infinity reaches the next transfer.
static const float MaxHalf = 65504.0;

void packVertexDifferential(out uint words[9], VertexDifferential differential)
{
    const float3 values[6] = { differential.positionDx, differential.positionDy,
        differential.normalDx, differential.normalDy,
        differential.directionDx, differential.directionDy };
    [unroll]
    for (uint i = 0u; i < 9u; ++i)
    {
        const uint first = i * 2u;
        const uint second = first + 1u;
        words[i] = packHalf2(clamp(values[first / 3u][first % 3u], -MaxHalf, MaxHalf),
            clamp(values[second / 3u][second % 3u], -MaxHalf, MaxHalf));
    }
}

VertexDifferential unpackVertexDifferential(uint words[9])
{
    float3 values[6];
    [unroll]
    for (uint i = 0u; i < 9u; ++i)
    {
        const float2 pair = unpackHalf2(words[i]);
        const uint first = i * 2u;
        const uint second = first + 1u;
        values[first / 3u][first % 3u] = pair.x;
        values[second / 3u][second % 3u] = pair.y;
    }
    VertexDifferential differential;
    differential.positionDx = values[0];
    differential.positionDy = values[1];
    differential.normalDx = values[2];
    differential.normalDy = values[3];
    differential.directionDx = values[4];
    differential.directionDy = values[5];
    return differential;
}

// The payload's fields, decoded.
struct RealtimeHit
{
    bool hit;
    uint instance;
    float t;
    float3 position;
    float3 geometricNormal;
    // The closure's shading normal, or the surface's where its lobes carry none.
    float3 normal;
    RealtimeClosure closure;
    float3 emission;
    float opacity;
};

void packRealtimeHit(inout RealtimeHitPayload payload, uint instance, float t, float3 position,
    float3 geometricNormal, float3 normal, RealtimeClosure closure, float3 emission)
{
    payload.instance = instance;
    payload.t = t;
    payload.position = position;
    payload.geometricNormal = packUnitVector(geometricNormal);
    payload.normal = packUnitVector(normal);
    packRealtimeClosure(payload.closure, closure);
    payload.emission = emission;
}

RealtimeHit unpackRealtimeHit(RealtimeHitPayload payload)
{
    RealtimeHit hit;
    hit.hit = payload.instance != RealtimeMissInstance;
    hit.instance = payload.instance;
    hit.t = payload.t;
    hit.position = payload.position;
    hit.geometricNormal = unpackUnitVector(payload.geometricNormal);
    hit.normal = unpackUnitVector(payload.normal);
    hit.closure = unpackRealtimeClosure(payload.closure);
    hit.emission = payload.emission;
    hit.opacity = payload.opacity;
    return hit;
}
