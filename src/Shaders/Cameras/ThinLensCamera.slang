#ifndef THINLENS_CAMERA_SLANG
#define THINLENS_CAMERA_SLANG

#include "../Core/RenderPolicy.slang"
#include "ICamera.slang"

// Thin-lens (perspective + depth-of-field) camera.
// Applies aperture sampling and focus-point projection on top of a standard
// perspective ray. DOF is disabled during interactive frames to keep preview
// snappy.
struct ThinLensCamera<TSampler> : ICamera<TSampler>
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
        CameraData camera = pushConstants.camera;
        float2 jitter = deterministicSample ? float2(0.0) : (sampler.get2D(samplerState) - 0.5);
        perspectiveBaseRay(pixelCoord, screenSize, jitter, rayOrigin, rayDirection);

        if (!deterministicSample && !isInteractiveFrame() && camera.aperture > 1e-6) {
            float2 lensSample = bokehDisk(sampler.get2D(samplerState), camera.bokehBias) * camera.aperture;
            float3 lensU      = normalize(camera.horizontal);
            float3 lensV      = normalize(camera.vertical);
            float3 focusPoint = rayOrigin + rayDirection * camera.focusDistance;
            rayOrigin    = camera.position + lensU * lensSample.x + lensV * lensSample.y;
            rayDirection = normalize(focusPoint - rayOrigin);
        }
    }
};

#endif // THINLENS_CAMERA_SLANG
