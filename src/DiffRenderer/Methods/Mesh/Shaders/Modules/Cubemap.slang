#pragma once

#include "EnvironmentLight.h"

typealias Light = EnvironmentLight::Light;

static const uint cubeFaces = 6;
static const float pi = 3.14159265358979;
static const uint rimPoints = 32;

uint getLevelSize(Light light, uint level) {
    return light.baseSize >> level;
}

uint getLevelOffset(Light light, uint level) {
    uint offset = 0;
    for (uint index = 0; index < level; ++index)
        offset += cubeFaces * getLevelSize(light, index) * getLevelSize(light, index);
    return offset;
}

uint getLevelTexelCount(Light light, uint level) {
    return cubeFaces * getLevelSize(light, level) * getLevelSize(light, level);
}

float getLevelRoughness(uint level) {
    if (level + 1 == EnvironmentLight::levelCount)
        return 1.0;
    return EnvironmentLight::minimumRoughness
           + (EnvironmentLight::maximumRoughness - EnvironmentLight::minimumRoughness) * float(level) / float(EnvironmentLight::levelCount - 2);
}

float getLevelCutoff(Light light, uint level) {
    return level < 4 ? light.cutoffLow[level] : light.cutoffHigh[level - 4];
}

//! The unnormalized direction of the face coordinates f in [-1, 1]^2, as nvdiffrec maps the faces; f may leave the face.
float3 getFaceVector(uint face, float2 f) {
    switch (face) {
    case 0: return float3(1.0, -f.y, -f.x);
    case 1: return float3(-1.0, -f.y, f.x);
    case 2: return float3(f.x, 1.0, f.y);
    case 3: return float3(f.x, -1.0, -f.y);
    case 4: return float3(f.x, -f.y, 1.0);
    default: return float3(-f.x, -f.y, -1.0);
    }
}

//! The direction in the frame of the face, with the face at z = 1.
[Differentiable]
float3 toFaceFrame(uint face, float3 v) {
    switch (face) {
    case 0: return float3(-v.z, -v.y, v.x);
    case 1: return float3(v.z, -v.y, -v.x);
    case 2: return float3(v.x, v.z, v.y);
    case 3: return float3(v.x, -v.z, -v.y);
    case 4: return float3(v.x, -v.y, v.z);
    default: return float3(-v.x, -v.y, -v.z);
    }
}

uint getFace(float3 v) {
    const float3 a = abs(v);
    if (a.x >= a.y && a.x >= a.z)
        return v.x >= 0.0 ? 0 : 1;
    if (a.y >= a.z)
        return v.y >= 0.0 ? 2 : 3;
    return v.z >= 0.0 ? 4 : 5;
}

float3 getTexelDirection(uint face, uint2 texel, uint size) {
    return normalize(getFaceVector(face, (float2(texel) + 0.5) / float(size) * 2.0 - 1.0));
}

//! The solid angle of a texel up to a constant, as pixel_area of nvdiffrec.
float getTexelArea(uint2 texel, uint size) {
    if (size <= 1)
        return 1.0;
    const int half = int(size / 2);
    const float2 d = float2(abs(int2(texel) - half));
    const float dx = atan((d.x + 1.0) / float(half)) - atan(d.x / float(half));
    const float dy = atan((d.y + 1.0) / float(half)) - atan(d.y / float(half));
    return dx * dy;
}

uint getTexelIndex(uint face, uint2 texel, uint size) {
    return (face * size + texel.y) * size + texel.x;
}

//! The continuous texel coordinates of a direction on a face, texel centers at integers.
[Differentiable]
float2 getFaceCoordinate(float3 v, no_diff uint face, no_diff uint size) {
    const float3 c = toFaceFrame(face, v);
    return (c.xy / c.z + 1.0) * 0.5 * float(size) - 0.5;
}

//! The texel that a tap outside the face reads: the nearest texel of the face that its direction falls on.
uint getSeamlessTexel(uint face, int2 texel, uint size) {
    if (all(texel >= 0) && all(texel < int(size)))
        return getTexelIndex(face, uint2(texel), size);
    const float3 v = getFaceVector(face, (float2(texel) + 0.5) / float(size) * 2.0 - 1.0);
    const uint other = getFace(v);
    const int2 nearest = clamp(int2(floor(getFaceCoordinate(v, other, size) + 0.5)), int2(0), int2(int(size) - 1));
    return getTexelIndex(other, uint2(nearest), size);
}

struct CubeTaps {
    uint4 texels;
    float2 fraction;
    uint face;
}

CubeTaps getCubeTaps(float3 direction, uint size) {
    CubeTaps taps;
    taps.face = getFace(direction);
    const float2 coordinate = getFaceCoordinate(direction, taps.face, size);
    const int2 base = int2(floor(coordinate));
    taps.fraction = coordinate - float2(base);
    taps.texels = uint4(getSeamlessTexel(taps.face, base, size), getSeamlessTexel(taps.face, base + int2(1, 0), size),
                        getSeamlessTexel(taps.face, base + int2(0, 1), size), getSeamlessTexel(taps.face, base + int2(1, 1), size));
    return taps;
}

float4 getTapWeights(float2 f) {
    return float4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
}

//! The texel rectangle of a face that holds every direction within the cone, or an empty one; the projected cone is convex, so its rim bounds it.
int4 getConeBounds(float3 axis, float cosine, uint face, uint size) {
    const int4 empty = int4(1, 0, 1, 0);
    const int4 full = int4(0, int(size) - 1, 0, int(size) - 1);
    const float3 c = toFaceFrame(face, axis);
    const float sine = sqrt(max(0.0, 1.0 - cosine * cosine));
    const float3 helper = abs(c.x) < 0.9 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
    const float3 u = normalize(cross(c, helper));
    const float3 w = cross(c, u);
    if (acos(clamp(c.z, -1.0, 1.0)) - acos(clamp(cosine, -1.0, 1.0)) >= 0.5 * pi)
        return empty;
    if (cosine <= 0.0)
        return full;
    float2 low = float2(1e30);
    float2 high = float2(-1e30);
    for (uint index = 0; index < rimPoints; ++index) {
        const float angle = 2.0 * pi * float(index) / float(rimPoints);
        const float3 rim = c * cosine + (u * cos(angle) + w * sin(angle)) * sine;
        if (rim.z <= 1e-4)
            return full;
        const float2 projected = rim.xy / rim.z;
        low = min(low, projected);
        high = max(high, projected);
    }
    if (any(low > 1.0) || any(high < -1.0))
        return empty;
    const int2 lowTexel = clamp(int2(floor((low + 1.0) * 0.5 * float(size) - 0.5)) - 1, int2(0), int2(int(size) - 1));
    const int2 highTexel = clamp(int2(ceil((high + 1.0) * 0.5 * float(size) - 0.5)) + 1, int2(0), int2(int(size) - 1));
    return int4(lowTexel.x, highTexel.x, lowTexel.y, highTexel.y);
}

float getGgx(float alphaSquared, float cosine) {
    const float c = clamp(cosine, 0.0, 1.0);
    const float d = (c * alphaSquared - c) * c + 1.0;
    return alphaSquared / (d * d * pi);
}

//! The weight of the source direction in the GGX filter around the axis, as specular_cubemap of nvdiffrec.
float getSpecularWeight(float3 axis, float3 source, float alphaSquared, float area) {
    const float3 h = normalize(source + axis);
    return max(dot(source, axis), 0.0) * getGgx(alphaSquared, max(dot(axis, h), 0.0)) * area / 4.0;
}

float getDiffuseWeight(float3 normal, float3 source, float area) {
    return min(max(dot(normal, source), 0.0), 0.999) * area / 3.141592;
}
