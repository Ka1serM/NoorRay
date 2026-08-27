#pragma once

// Scattering closure results, a 1:1 port of Materials/Shading/BsdfClosure.h
// and Rendering/Sampling/HemisphereSampler.h.

#include "Spectrum.slang"

static const float BsdfEpsilon = 1.0e-5;
static const float BsdfPi = 3.14159265358979323846;
static const float BsdfSingularPdf = 1.0e6;

static const uint BsdfEventDiffuse = 0u;
static const uint BsdfEventSpecular = 1u;
static const uint BsdfEventTransmission = 2u;

// All scattering closures return their value and sampling density together, so
// callers cannot mix an evaluation from one path with a PDF from another.
struct BsdfEvaluation
{
    SampledSpectrum value;
    float pdf;

    bool isFinite() { return spectrumIsFinite(value) && !isnan(pdf) && !isinf(pdf); }
};

BsdfEvaluation makeBsdfEvaluation()
{
    BsdfEvaluation result;
    result.value = SampledSpectrum(0.0);
    result.pdf = 0.0;
    return result;
}

struct BsdfSample
{
    float3 direction;
    SampledSpectrum weight;
    float pdf;
    float eta;
    uint event;
    bool singular;
    // A Sellmeier transmission uses the hero wavelength for this packet's
    // direction. The caller terminates secondary wavelengths before tracing the
    // next segment, so later path samples carry their own refracted directions.
    bool dispersive;

    bool directionIsValid()
    {
        const float lengthSquared = dot(direction, direction);
        return !any(isnan(direction)) && !any(isinf(direction))
            && !isnan(lengthSquared) && !isinf(lengthSquared)
            && lengthSquared > 1.0e-12
            && !isnan(pdf) && !isinf(pdf) && pdf >= 0.0
            && spectrumIsFinite(weight);
    }
};

BsdfSample makeBsdfSample()
{
    BsdfSample result;
    result.direction = float3(0.0);
    result.weight = SampledSpectrum(0.0);
    result.pdf = 0.0;
    result.eta = 1.0;
    result.event = BsdfEventDiffuse;
    result.singular = false;
    result.dispersive = false;
    return result;
}

// ---------------------------------------------------------------------------
// Hemisphere sampling (HemisphereSampler.h)
// ---------------------------------------------------------------------------

// Cycles' concentric square-to-disk mapping. Compared with the usual polar
// mapping it preserves the stratification of Owen-Sobol and other structured
// 2D sample sequences, which matters for both BSDF and light sampling.
float2 concentricDisk(float2 sample)
{
    const float a = 2.0 * sample.x - 1.0;
    const float b = 2.0 * sample.y - 1.0;
    if (a == 0.0 && b == 0.0)
        return float2(0.0);

    float radius;
    float phi;
    if (a * a > b * b)
    {
        radius = a;
        phi = 0.25 * BsdfPi * (b / a);
    }
    else
    {
        radius = b;
        phi = 0.5 * BsdfPi - 0.25 * BsdfPi * (a / b);
    }
    return float2(radius * cos(phi), radius * sin(phi));
}

void buildBasis(float3 normal, out float3 tangent, out float3 bitangent)
{
    const float3 helper = abs(normal.z) < 0.999
        ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    tangent = normalize(cross(normal, helper));
    bitangent = cross(tangent, normal);
}

float3 uniformHemisphere(float2 sample)
{
    float2 disk = concentricDisk(sample);
    const float z = 1.0 - dot(disk, disk);
    disk *= sqrt(max(z + 1.0, 0.0));
    return float3(disk.x, disk.y, z);
}

float3 cosineHemisphere(float3 normal, float2 sample)
{
    const float2 disk = concentricDisk(sample);
    float3 tangent, bitangent;
    buildBasis(normal, tangent, bitangent);
    return tangent * disk.x + bitangent * disk.y
        + normal * sqrt(max(0.0, 1.0 - dot(disk, disk)));
}
