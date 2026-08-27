#pragma once

// Jakob & Hanika RGB-to-spectrum upsampling, matching
// Materials/Shading/RgbToSpectrum.h and pbrt-v4's 64^3 sRGB table. The table
// follows CIE/D65 in the immutable spectral-table ByteAddressBuffer.

#include "Spectrum.slang"

static const int RgbSpectrumResolution = 64;
static const uint RgbSpectrumScaleOffset = SpectralTableFloatCount;
static const uint RgbSpectrumCoefficientOffset =
    RgbSpectrumScaleOffset + uint(RgbSpectrumResolution);
static const uint RgbSpectrumCoefficientStride3 =
    uint(RgbSpectrumResolution * RgbSpectrumResolution
        * RgbSpectrumResolution * 3);
static const uint RgbSpectrumCoefficientStride2 =
    uint(RgbSpectrumResolution * RgbSpectrumResolution * 3);
static const uint RgbSpectrumCoefficientStride1 =
    uint(RgbSpectrumResolution * 3);

float rgbSpectrumTableFloat(ByteAddressBuffer tables, uint offset)
{
    return asfloat(tables.Load(offset * 4u));
}

float rgbSpectrumSigmoid(float x)
{
    if (x > 1.0e10)
        return 1.0;
    if (x < -1.0e10)
        return 0.0;
    return 0.5 + x * rsqrt(4.0 + 4.0 * x * x);
}

float rgbSpectrumSigmoidEval(float3 coefficients, float wavelength)
{
    return rgbSpectrumSigmoid(coefficients.z
        + wavelength * (coefficients.y + wavelength * coefficients.x));
}

int rgbSpectrumFindInterval(ByteAddressBuffer tables, float z)
{
    int lo = 0;
    int hi = RgbSpectrumResolution - 2;
    [loop]
    while (lo < hi)
    {
        const int mid = (lo + hi + 1) >> 1;
        if (rgbSpectrumTableFloat(tables,
                RgbSpectrumScaleOffset + uint(mid)) < z)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

float3 rgbSpectrumLookupCoefficients(ByteAddressBuffer tables, float3 rgb)
{
    rgb = clamp(rgb, 0.0, 1.0);
    if (rgb.x == rgb.y && rgb.y == rgb.z)
    {
        const float denominator = rgb.x * (1.0 - rgb.x);
        const float c2 = denominator < 1.0e-10
            ? (rgb.x <= 0.5 ? -1.0e30 : 1.0e30)
            : (rgb.x - 0.5) / sqrt(denominator);
        return float3(0.0, 0.0, c2);
    }

    const int dominant = rgb.x > rgb.y
        ? (rgb.x > rgb.z ? 0 : 2)
        : (rgb.y > rgb.z ? 1 : 2);
    const float z = dominant == 0 ? rgb.x : dominant == 1 ? rgb.y : rgb.z;
    const float xComponent = dominant == 0 ? rgb.y
        : dominant == 1 ? rgb.z : rgb.x;
    const float yComponent = dominant == 0 ? rgb.z
        : dominant == 1 ? rgb.x : rgb.y;
    const float x = xComponent * float(RgbSpectrumResolution - 1) / z;
    const float y = yComponent * float(RgbSpectrumResolution - 1) / z;
    const int xi = min(int(x), RgbSpectrumResolution - 2);
    const int yi = min(int(y), RgbSpectrumResolution - 2);
    const int zi = rgbSpectrumFindInterval(tables, z);
    const float dx = x - float(xi);
    const float dy = y - float(yi);
    const float scale0 = rgbSpectrumTableFloat(tables,
        RgbSpectrumScaleOffset + uint(zi));
    const float scale1 = rgbSpectrumTableFloat(tables,
        RgbSpectrumScaleOffset + uint(zi + 1));
    const float dz = (z - scale0) / (scale1 - scale0);
    const uint base = RgbSpectrumCoefficientOffset
        + uint(dominant) * RgbSpectrumCoefficientStride3;

    float3 corners[8];
    [unroll]
    for (int az = 0; az < 2; ++az)
    [unroll]
    for (int ay = 0; ay < 2; ++ay)
    [unroll]
    for (int ax = 0; ax < 2; ++ax)
    {
        const uint address = base + uint(zi + az) * RgbSpectrumCoefficientStride2
            + uint(yi + ay) * RgbSpectrumCoefficientStride1
            + uint(xi + ax) * 3u;
        corners[az * 4 + ay * 2 + ax] = float3(
            rgbSpectrumTableFloat(tables, address),
            rgbSpectrumTableFloat(tables, address + 1u),
            rgbSpectrumTableFloat(tables, address + 2u));
    }
    const float3 z0 = lerp(lerp(corners[0], corners[1], dx),
        lerp(corners[2], corners[3], dx), dy);
    const float3 z1 = lerp(lerp(corners[4], corners[5], dx),
        lerp(corners[6], corners[7], dx), dy);
    return lerp(z0, z1, dz);
}

SampledSpectrum rgbAlbedoToSpectrum(ByteAddressBuffer tables, float3 rgb,
    SampledWavelengths wavelengths)
{
    const float3 coefficients = rgbSpectrumLookupCoefficients(tables, rgb);
    SampledSpectrum result;
    [unroll]
    for (int i = 0; i < NrSpectrumSamples; ++i)
        result[i] = rgbSpectrumSigmoidEval(coefficients, wavelengths.lambda[i]);
    return result;
}

SampledSpectrum rgbIlluminantToSpectrum(ByteAddressBuffer tables, float3 rgb,
    SampledWavelengths wavelengths)
{
    const float maximum = max(rgb.x, max(rgb.y, rgb.z));
    if (maximum <= 0.0)
        return SampledSpectrum(0.0);
    const float scale = 2.0 * maximum;
    const float3 coefficients = rgbSpectrumLookupCoefficients(tables, rgb / scale);
    SampledSpectrum result;
    [unroll]
    for (int i = 0; i < NrSpectrumSamples; ++i)
        result[i] = scale * rgbSpectrumSigmoidEval(coefficients,
            wavelengths.lambda[i]) * sampleD65(tables, wavelengths.lambda[i]);
    return result;
}
