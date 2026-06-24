#ifndef HALTON_SAMPLER_SLANG
#define HALTON_SAMPLER_SLANG

#include "../Core/Core.slang"
#include "ISampler.slang"

float haltonRadicalInverse(uint index, uint baseValue)
{
    float invBase = 1.0 / float(baseValue);
    float invBasePow = invBase;
    float value = 0.0;
    uint current = index;

    while (current > 0u) {
        uint digit = current % baseValue;
        value += float(digit) * invBasePow;
        current /= baseValue;
        invBasePow *= invBase;
    }

    return value;
}

float haltonForDimension(uint index, int dimension)
{
    if (dimension == 0)
        return haltonRadicalInverse(index, 2u);
    if (dimension == 1)
        return haltonRadicalInverse(index, 3u);
    if (dimension == 2)
        return haltonRadicalInverse(index, 5u);
    if (dimension == 3)
        return haltonRadicalInverse(index, 7u);
    if (dimension == 4)
        return haltonRadicalInverse(index, 11u);
    return haltonRadicalInverse(index, 13u);
}

struct HaltonSampler : ISampler
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
        uint sampleIndex = uint(max(samplerState.frame + 1, 1));
        float value = haltonForDimension(sampleIndex, samplerState.dimension);
        float scramble = (samplerState.dimension & 1) == 0 ? samplerState.randomOffset.x : samplerState.randomOffset.y;
        samplerState.dimension += 1;
        return frac(value + scramble);
    }

    float2 get2D(inout SamplerState samplerState)
    {
        float x = get1D(samplerState);
        float y = get1D(samplerState);
        return float2(x, y);
    }
};

#endif // HALTON_SAMPLER_SLANG
