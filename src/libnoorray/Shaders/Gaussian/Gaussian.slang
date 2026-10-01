#pragma once

#include "../Raytracer/ByteBuffer.slang"



// Gaussian splat shading shared by the splat hit stages. A splat is an
// instance of its cloud's proxy mesh; its opacity and spherical harmonics
// live in the parameters of the cloud's splat material (GaussianCloud),
// read here by byte address:
//   header:  cutoff distance squared (float), splat count, two unused words
//   entry i: opacity (float) @ 0, SH coefficient count @ 4, 48 half SH @ 8

static const uint GaussianHeaderSize = 16;
static const uint GaussianRecordStride = 104;
static const uint GaussianOpacityOffset = 0;
static const uint GaussianShCountOffset = 4;
static const uint GaussianShOffset = 8;
static const uint GaussianShChannelCount = 3;
static const uint GaussianMaxShCoefficients = 16;
static const float GaussianInv4Pi = 0.07957747154594767;
static const float GaussianShC0 = 0.28209479177387814;

struct GaussianHit
{
    uint id;
    float t;
    float distanceSq;
    float acceptance;
    bool valid;
};

// Keep random generation behind an interface so Owen/Sobol and deterministic
// replay samplers can share exactly the same acceptance path.
interface IGaussianSampler
{
    float sample1D(uint dimension);
};

struct GaussianHashSampler : IGaussianSampler
{
    uint seed;

    uint hash(uint value)
    {
        value ^= value >> 16;
        value *= 0x7feb352du;
        value ^= value >> 15;
        value *= 0x846ca68bu;
        return value ^ (value >> 16);
    }

    float sample1D(uint dimension)
    {
        return (float(hash(seed ^ (dimension * 0x9e3779b9u)) & 0x00ffffffu)
            + 0.5) / 16777216.0;
    }
};

uint gaussianBase(uint gaussianId)
{
    return GaussianHeaderSize + gaussianId * GaussianRecordStride;
}

float gaussianCutoffDistanceSq(ByteBuffer records)
{
    return asfloat(records.Load(0u));
}

float gaussianFloat(ByteBuffer records, uint gaussianId, uint offset)
{
    return asfloat(records.Load(gaussianBase(gaussianId) + offset));
}

uint gaussianUint(ByteBuffer records, uint gaussianId, uint offset)
{
    return records.Load(gaussianBase(gaussianId) + offset);
}

// Portable IEEE-754 binary16 decode for devices without native half storage.
float gaussianHalfToFloat(uint bits)
{
    const uint sign = (bits >> 15) & 1u;
    const uint exponent = (bits >> 10) & 31u;
    const uint mantissa = bits & 1023u;
    if (exponent == 0u)
    {
        if (mantissa == 0u)
            return sign != 0u ? -0.0 : 0.0;
        // binary16 subnormals are mantissa * 2^-24.
        const float value = float(mantissa) * 0.000000059604644775390625;
        return sign != 0u ? -value : value;
    }
    if (exponent == 31u)
        return mantissa == 0u ? (sign != 0u ? -asfloat(0x7f800000u)
                                            : asfloat(0x7f800000u))
                              : asfloat(0x7fc00000u);
    const float value = (1.0 + float(mantissa) / 1024.0)
        * exp2(float(exponent) - 15.0);
    return sign != 0u ? -value : value;
}

float gaussianHalf(ByteBuffer records, uint gaussianId, uint index)
{
    const uint address = gaussianBase(gaussianId) + GaussianShOffset
        + index * 2u;
    const uint word = records.Load(address & ~3u);
    const uint bits = (address & 2u) != 0u ? word >> 16 : word & 0xffffu;
    return gaussianHalfToFloat(bits);
}

float3 gaussianCoefficient(ByteBuffer records, uint gaussianId,
    uint coefficient)
{
    if (coefficient >= min(gaussianUint(records, gaussianId,
        GaussianShCountOffset), GaussianMaxShCoefficients))
        return float3(0.0);
    const uint base = coefficient * GaussianShChannelCount;
    return float3(gaussianHalf(records, gaussianId, base),
        gaussianHalf(records, gaussianId, base + 1u),
        gaussianHalf(records, gaussianId, base + 2u));
}

float3 evaluateGaussianSh(ByteBuffer records, uint gaussianId,
    uint order, float3 viewDirection)
{
    const float3 direction = normalize(viewDirection);
    const float x = direction.x;
    const float y = direction.y;
    const float z = direction.z;
    float3 result = GaussianShC0 * gaussianCoefficient(records, gaussianId, 0u);
    const float C1 = 0.4886025119029199;
    const float C2[5] = { 1.0925484305920792, -1.0925484305920792,
        0.31539156525252005, -1.0925484305920792, 0.5462742152960396 };
    const float C3[7] = { -0.5900435899266435, 2.890611442640554,
        -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
        1.445305721320277, -0.5900435899266435 };
    if (order >= 1u)
        result += -C1 * y * gaussianCoefficient(records, gaussianId, 1u)
            + C1 * z * gaussianCoefficient(records, gaussianId, 2u)
            - C1 * x * gaussianCoefficient(records, gaussianId, 3u);
    if (order >= 2u)
        result += C2[0] * x * y * gaussianCoefficient(records, gaussianId, 4u)
            + C2[1] * y * z * gaussianCoefficient(records, gaussianId, 5u)
            + C2[2] * (2.0 * z * z - x * x - y * y)
                * gaussianCoefficient(records, gaussianId, 6u)
            + C2[3] * x * z * gaussianCoefficient(records, gaussianId, 7u)
            + C2[4] * (x * x - y * y) * gaussianCoefficient(records, gaussianId, 8u);
    if (order >= 3u)
        result += C3[0] * y * (3.0 * x * x - y * y)
                * gaussianCoefficient(records, gaussianId, 9u)
            + C3[1] * x * y * z * gaussianCoefficient(records, gaussianId, 10u)
            + C3[2] * y * (4.0 * z * z - x * x - y * y)
                * gaussianCoefficient(records, gaussianId, 11u)
            + C3[3] * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)
                * gaussianCoefficient(records, gaussianId, 12u)
            + C3[4] * x * (4.0 * z * z - x * x - y * y)
                * gaussianCoefficient(records, gaussianId, 13u)
            + C3[5] * z * (x * x - y * y) * gaussianCoefficient(records, gaussianId, 14u)
            + C3[6] * x * (x * x - 3.0 * y * y)
                * gaussianCoefficient(records, gaussianId, 15u);
    return float3(0.5) + result;
}

float gaussianDistanceSq(float3 origin, float3 direction, float tMin,
    out float hitT)
{
    const float directionSq = dot(direction, direction);
    const float projectedT = directionSq > 0.0
        ? -dot(origin, direction) / directionSq : 0.0;
    hitT = max(projectedT, tMin);
    const float3 hit = fma(direction, hitT, origin);
    return dot(hit, hit);
}

// This is the direct equivalent of __anyhit__gaussian_path's acceptance
// predicate.  The caller supplies the proxy-space ray and the current path
// t-max; closest-hit can then use the returned distance for shading.
GaussianHit acceptGaussian<S : IGaussianSampler>(S sampler, uint gaussianId,
    float opacity, float3 origin, float3 direction, float tMin, float tMax,
    float cutoffDistanceSq, uint sampleDimension)
{
    GaussianHit result;
    result.id = gaussianId;
    result.t = 0.0;
    result.distanceSq = 0.0;
    result.acceptance = 0.0;
    result.valid = false;
    if (opacity <= 0.0)
        return result;
    const float xi = sampler.sample1D(sampleDimension);
    result.acceptance = xi;
    if (xi >= opacity)
        return result;
    result.distanceSq = gaussianDistanceSq(origin, direction, tMin, result.t);
    if (isnan(result.t) || isinf(result.t) || result.t >= tMax
        || result.distanceSq >= cutoffDistanceSq
        || xi >= opacity * exp(-0.5 * result.distanceSq))
        return result;
    result.valid = true;
    return result;
}

GaussianHit sampleGaussian<S : IGaussianSampler>(S sampler,
    ByteBuffer records, uint gaussianId,
    float3 origin, float3 direction, float tMin, float tMax,
    float cutoffDistanceSq, uint sampleDimension)
{
    return acceptGaussian(sampler, gaussianId,
        gaussianFloat(records, gaussianId, GaussianOpacityOffset), origin,
        direction, tMin, tMax, cutoffDistanceSq, sampleDimension);
}
