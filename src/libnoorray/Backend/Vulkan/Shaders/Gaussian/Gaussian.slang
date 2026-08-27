#pragma once

// Gaussian proxy shading shared by all Vulkan path entry points.  The record
// accessors deliberately use byte addresses: Gaussian is a host/GPU POD whose
// layout is owned by GaussianAsset (transform[12] @ 0, opacity @ 48, packed
// half SH @ 52, count @ 148).  This keeps Slang's matrix packing conventions
// out of the scene ABI.

static const uint GaussianRecordStride = 152;
static const uint GaussianOpacityOffset = 48;
static const uint GaussianShOffset = 52;
static const uint GaussianShCountOffset = 148;
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
    return gaussianId * GaussianRecordStride;
}

float gaussianFloat(ByteAddressBuffer records, uint gaussianId, uint offset)
{
    return asfloat(records.Load(gaussianBase(gaussianId) + offset));
}

uint gaussianUint(ByteAddressBuffer records, uint gaussianId, uint offset)
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

float gaussianHalf(ByteAddressBuffer records, uint gaussianId, uint index)
{
    const uint address = gaussianBase(gaussianId) + GaussianShOffset
        + index * 2u;
    const uint word = records.Load(address & ~3u);
    const uint bits = (address & 2u) != 0u ? word >> 16 : word & 0xffffu;
    return gaussianHalfToFloat(bits);
}

float3 gaussianCoefficient(ByteAddressBuffer records, uint gaussianId,
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

float3 evaluateGaussianSh(ByteAddressBuffer records, uint gaussianId,
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

// Scene::buildGaussianRenderData() also exposes a compact SoA form for large
// splat sets.  Keep its addressing separate from GaussianAsset's 152-byte
// records; both forms feed the same SH basis and acceptance routines.
float gaussianPackedHalf(ByteAddressBuffer coefficients, uint gaussianId,
    uint coefficientCount, uint index)
{
    const uint address = (gaussianId * coefficientCount
        * GaussianShChannelCount + index) * 2u;
    const uint word = coefficients.Load(address & ~3u);
    return gaussianHalfToFloat((address & 2u) != 0u ? word >> 16 : word & 0xffffu);
}

float3 gaussianPackedCoefficient(ByteAddressBuffer coefficients, uint gaussianId,
    uint coefficientCount, uint coefficient)
{
    if (coefficient >= coefficientCount)
        return float3(0.0);
    const uint base = coefficient * GaussianShChannelCount;
    return float3(gaussianPackedHalf(coefficients, gaussianId, coefficientCount, base),
        gaussianPackedHalf(coefficients, gaussianId, coefficientCount, base + 1u),
        gaussianPackedHalf(coefficients, gaussianId, coefficientCount, base + 2u));
}

float3 evaluateGaussianPackedSh(ByteAddressBuffer coefficients, uint gaussianId,
    uint coefficientCount, uint order, float3 viewDirection)
{
    const float3 direction = normalize(viewDirection);
    const float x = direction.x;
    const float y = direction.y;
    const float z = direction.z;
    float3 result = GaussianShC0 * gaussianPackedCoefficient(
        coefficients, gaussianId, coefficientCount, 0u);
    const float C1 = 0.4886025119029199;
    const float C2[5] = { 1.0925484305920792, -1.0925484305920792,
        0.31539156525252005, -1.0925484305920792, 0.5462742152960396 };
    const float C3[7] = { -0.5900435899266435, 2.890611442640554,
        -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
        1.445305721320277, -0.5900435899266435 };
    if (order >= 1u)
        result += -C1 * y * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 1u)
            + C1 * z * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 2u)
            - C1 * x * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 3u);
    if (order >= 2u)
        result += C2[0] * x * y * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 4u)
            + C2[1] * y * z * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 5u)
            + C2[2] * (2.0 * z * z - x * x - y * y)
                * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 6u)
            + C2[3] * x * z * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 7u)
            + C2[4] * (x * x - y * y)
                * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 8u);
    if (order >= 3u)
        result += C3[0] * y * (3.0 * x * x - y * y)
                * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 9u)
            + C3[1] * x * y * z * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 10u)
            + C3[2] * y * (4.0 * z * z - x * x - y * y)
                * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 11u)
            + C3[3] * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)
                * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 12u)
            + C3[4] * x * (4.0 * z * z - x * x - y * y)
                * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 13u)
            + C3[5] * z * (x * x - y * y)
                * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 14u)
            + C3[6] * x * (x * x - 3.0 * y * y)
                * gaussianPackedCoefficient(coefficients, gaussianId, coefficientCount, 15u);
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
    ByteAddressBuffer records, uint gaussianId,
    float3 origin, float3 direction, float tMin, float tMax,
    float cutoffDistanceSq, uint sampleDimension)
{
    return acceptGaussian(sampler, gaussianId,
        gaussianFloat(records, gaussianId, GaussianOpacityOffset), origin,
        direction, tMin, tMax, cutoffDistanceSq, sampleDimension);
}

GaussianHit samplePackedGaussian<S : IGaussianSampler>(S sampler,
    ByteAddressBuffer opacities, uint gaussianId, float3 origin,
    float3 direction, float tMin, float tMax, float cutoffDistanceSq,
    uint sampleDimension)
{
    return acceptGaussian(sampler, gaussianId,
        asfloat(opacities.Load(gaussianId * 4u)), origin, direction,
        tMin, tMax, cutoffDistanceSq, sampleDimension);
}

float3 sampleGaussianIsotropic(uint seed, uint dimension)
{
    GaussianHashSampler sampler;
    sampler.seed = seed;
    const float theta = 6.283185307179586 * sampler.sample1D(dimension);
    const float z = 2.0 * sampler.sample1D(dimension + 1u) - 1.0;
    const float radius = sqrt(max(1.0 - z * z, 0.0));
    return float3(radius * cos(theta), z, radius * sin(theta));
}

// GaussianAsset records contain the object-to-world R*S columns followed by
// the center. Projecting onto each column gives the exact inverse for that
// diagonalized representation (including anisotropic scale) without relying
// on Slang's matrix packing rules.
float3 gaussianRecordVector(ByteAddressBuffer records, uint gaussianId,
    uint column)
{
    const uint address = gaussianBase(gaussianId) + column * 12u;
    return float3(asfloat(records.Load(address)),
        asfloat(records.Load(address + 4u)),
        asfloat(records.Load(address + 8u)));
}

float3 gaussianRecordCenter(ByteAddressBuffer records, uint gaussianId)
{
    return gaussianRecordVector(records, gaussianId, 3u);
}

struct GaussianTraceResult
{
    uint id;
    float t;
    float distanceSq;
    float opacity;
    float3 localPoint;
    float3 normal;
    bool valid;
};

// A flat-record fallback has no spatial index. Keep the work per pixel
// bounded so a malformed or very large splat asset cannot monopolize a
// dispatch. Scenes above this limit are visited at a deterministic stride;
// the proxy-AS path can later replace this candidate policy without changing
// the acceptance or shading contracts.
static const uint GaussianBruteForceLimit = 4096u;

// Native Vulkan fallback for Gaussian proxies.  It uses the same normalized
// metric and stochastic acceptance as the proxy any-hit path.  Keeping this
// routine separate lets a future proxy acceleration structure replace only
// the candidate loop without changing shading or AOV code.
GaussianTraceResult traceGaussianRecords(ByteAddressBuffer records, uint count,
    float3 origin, float3 direction, float tMin, float tMax,
    float cutoffDistanceSq, uint seed)
{
    GaussianTraceResult result;
    result.id = 0u;
    result.t = tMax;
    result.distanceSq = 0.0;
    result.opacity = 0.0;
    result.localPoint = float3(0.0);
    result.normal = float3(0.0, 0.0, 1.0);
    result.valid = false;
    if (count == 0u)
        return result;
    GaussianHashSampler sampler;
    sampler.seed = seed;

    const uint visitCount = min(count, GaussianBruteForceLimit);
    const uint visitStride = max((count + visitCount - 1u) / visitCount, 1u);
    for (uint visit = 0u; visit < visitCount; ++visit)
    {
        const uint gaussianId = min(visit * visitStride, count - 1u);
        const float3 axis0 = gaussianRecordVector(records, gaussianId, 0u);
        const float3 axis1 = gaussianRecordVector(records, gaussianId, 1u);
        const float3 axis2 = gaussianRecordVector(records, gaussianId, 2u);
        const float axis0Sq = dot(axis0, axis0);
        const float axis1Sq = dot(axis1, axis1);
        const float axis2Sq = dot(axis2, axis2);
        if (axis0Sq <= 1.0e-12 || axis1Sq <= 1.0e-12 || axis2Sq <= 1.0e-12)
            continue;

        const float3 relative = origin - gaussianRecordCenter(records, gaussianId);
        const float3 localOrigin = float3(dot(relative, axis0) / axis0Sq,
            dot(relative, axis1) / axis1Sq, dot(relative, axis2) / axis2Sq);
        const float3 localDirection = float3(dot(direction, axis0) / axis0Sq,
            dot(direction, axis1) / axis1Sq, dot(direction, axis2) / axis2Sq);
        const float directionSq = dot(localDirection, localDirection);
        if (directionSq <= 1.0e-12)
            continue;
        const float projectedT = -dot(localOrigin, localDirection) / directionSq;
        const float t = max(projectedT, tMin);
        if (t >= result.t || t >= tMax)
            continue;
        const float3 localPoint = fma(localDirection, t, localOrigin);
        const float distanceSq = dot(localPoint, localPoint);
        if (distanceSq >= cutoffDistanceSq)
            continue;
        const float opacity = gaussianFloat(records, gaussianId,
            GaussianOpacityOffset);
        if (opacity <= 0.0)
            continue;
        const float xi = sampler.sample1D(gaussianId + 17u);
        if (xi >= opacity || xi >= opacity * exp(-0.5 * distanceSq))
            continue;

        result.id = gaussianId;
        result.t = t;
        result.distanceSq = distanceSq;
        result.opacity = opacity * exp(-0.5 * distanceSq);
        result.localPoint = localPoint;
        result.normal = normalize(localPoint.x * axis0 / axis0Sq
            + localPoint.y * axis1 / axis1Sq
            + localPoint.z * axis2 / axis2Sq);
        result.valid = true;
    }
    return result;
}
