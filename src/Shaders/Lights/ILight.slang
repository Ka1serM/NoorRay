#ifndef ILIGHT_SLANG
#define ILIGHT_SLANG

struct LightLiSample
{
    float3 wi;
    float3 Li;
    float pdf;
};

struct LightLeSample
{
    float3 direction;
    float3 radiance;
    float pdfPos;
    float pdfDir;
};

interface ILight
{
    LightLiSample sampleLi(float3 referencePoint, float2 u);
    float pdfLi(float3 referencePoint, float3 wi);
    LightLeSample sampleLe(float2 u1, float2 u2);
};

#endif // ILIGHT_SLANG
