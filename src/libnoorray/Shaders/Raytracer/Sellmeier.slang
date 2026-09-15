#pragma once

#include "Spectrum.slang"

float sellmeierIor(float3 b, float3 c, float wavelengthNm)
{
    const float wavelengthUm = wavelengthNm * 0.001;
    const float lambdaSquared = wavelengthUm * wavelengthUm;
    float nSquared = 1.0;
    [unroll]
    for (int i = 0; i < 3; ++i)
    {
        const float denominator = lambdaSquared - c[i];
        if (abs(denominator) > 1.0e-8)
            nSquared += b[i] * lambdaSquared / denominator;
    }
    return sqrt(max(nSquared, 1.0));
}

SampledSpectrum sellmeierSpectrum(float3 b, float3 c,
    SampledWavelengths wavelengths)
{
    SampledSpectrum result;
    [unroll]
    for (int i = 0; i < NrSpectrumSamples; ++i)
        result[i] = sellmeierIor(b, c, wavelengths.lambda[i]);
    return result;
}
