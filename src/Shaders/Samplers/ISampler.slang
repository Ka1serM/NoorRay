#ifndef ISAMPLER_SLANG
#define ISAMPLER_SLANG

struct SamplerState
{
    float2 randomOffset;
    int frame;
    int dimension;
};

interface ISampler
{
    void startPixel(int2 pixelCoord, int sampleIndex, out SamplerState samplerState);
    float get1D(inout SamplerState samplerState);
    float2 get2D(inout SamplerState samplerState);
};

#endif // ISAMPLER_SLANG
