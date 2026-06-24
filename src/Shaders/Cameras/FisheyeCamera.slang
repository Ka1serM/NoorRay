#ifndef FISHEYE_CAMERA_SLANG
#define FISHEYE_CAMERA_SLANG

#include "../Core/Core.slang"
#include "../Core/RenderPolicy.slang"
#include "ICamera.slang"

// Equidistant fisheye camera with optional thin-lens depth-of-field.
struct FisheyeCamera<TSampler> : ICamera<TSampler>
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
        float2 uv = deterministicSample
            ? (float2(pixelCoord) + 0.5) / float2(screenSize)
            : (float2(pixelCoord) + sampler.get2D(samplerState)) / float2(screenSize);

        float2 sensorOffset = float2(uv.x - 0.5, 0.5 - uv.y);
        float  aspect       = float(screenSize.x) / float(screenSize.y);
        float2 uvCentered   = float2(sensorOffset.x * aspect, sensorOffset.y) * 2.0;

        float r    = length(uvCentered);
        float maxR = length(float2(aspect, 1.0));

        float maxTheta = camera.fisheyeFov * 0.5;
        float theta    = min((r / maxR) * maxTheta, PI * 0.5);

        float3 forward = normalize(camera.direction);
        float3 right   = normalize(camera.horizontal);
        float3 up      = normalize(camera.vertical);

        rayOrigin = camera.position;
        if (r > 1e-6) {
            float phi = atan2(uvCentered.y, uvCentered.x);
            rayDirection = normalize(
                sin(theta) * cos(phi) * right +
                sin(theta) * sin(phi) * up    +
                cos(theta)            * forward);
        } else {
            rayDirection = forward;
        }

        if (!deterministicSample && !isInteractiveFrame() && camera.aperture > 1e-6) {
            float2 lensSample = bokehDisk(sampler.get2D(samplerState), camera.bokehBias) * camera.aperture;
            float3 focusPoint = rayOrigin + rayDirection * camera.focusDistance;
            rayOrigin    = camera.position + right * lensSample.x + up * lensSample.y;
            rayDirection = normalize(focusPoint - rayOrigin);
        }
    }
};

#endif // FISHEYE_CAMERA_SLANG
