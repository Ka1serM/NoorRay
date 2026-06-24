#ifndef PERSPECTIVE_CAMERA_SLANG
#define PERSPECTIVE_CAMERA_SLANG

#include "ICamera.slang"

// Pinhole perspective camera — no depth-of-field.
struct PerspectiveCamera<TSampler> : ICamera<TSampler>
    where TSampler : ISampler
{
    void generateRay(
        int2 pixelCoord,
        int2 screenSize,
        inout SamplerState samplerState,
        inout TSampler sampler,
        bool deterministicSample,
        out float3 rayOrigin,
        out float3 rayDirection)
    {
        float2 jitter = deterministicSample ? float2(0.0) : (sampler.get2D(samplerState) - 0.5);
        perspectiveBaseRay(pixelCoord, screenSize, jitter, rayOrigin, rayDirection);
    }
};

#endif // PERSPECTIVE_CAMERA_SLANG
