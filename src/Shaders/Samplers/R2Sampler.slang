#ifndef R2_SAMPLER_SLANG
#define R2_SAMPLER_SLANG

#include "../Core/Core.slang"
#include "ISampler.slang"

static const float R2_G = 1.32471795724;
static const float R2_G2 = R2_G * R2_G;
static const float R2_G3 = R2_G * R2_G2;
static const float R2_G4 = R2_G * R2_G3;
static const float R2_G5 = R2_G * R2_G4;
static const float R2_G6 = R2_G * R2_G5;

float2 r2AlphaForDimension(int dimension)
{
    if (dimension < 2)
        return float2(1.0 / R2_G, 1.0 / R2_G2);
    if (dimension < 4)
        return float2(1.0 / R2_G3, 1.0 / R2_G4);
    return float2(1.0 / R2_G5, 1.0 / R2_G6);
}

struct R2Sampler : ISampler
{
    void startPixel(int2 pixelCoord, int sampleIndex, out SamplerState samplerState)
    {
        uint2 hash = pcg2d(uint2(pixelCoord));
        samplerState.randomOffset = float2(hash) / float(0xFFFFFFFFu);
        samplerState.frame = sampleIndex;
        samplerState.dimension = 0;
    }

    float get1D(inout SamplerState samplerState)
    {
        float2 sample = get2D(samplerState);
        return sample.x;
    }

    float2 get2D(inout SamplerState samplerState)
    {
        float2 alpha = r2AlphaForDimension(samplerState.dimension);
        samplerState.dimension += 2;
        return fract(alpha * float(samplerState.frame) + samplerState.randomOffset);
    }
};

#endif // R2_SAMPLER_SLANG
