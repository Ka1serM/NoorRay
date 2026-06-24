#ifndef ORTHOGRAPHIC_CAMERA_SLANG
#define ORTHOGRAPHIC_CAMERA_SLANG

#include "ICamera.slang"

struct OrthographicCamera<TSampler> : ICamera<TSampler>
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

        // Horizontal and Vertical store half-extents of the orthographic view volume.
        // sensorOffset maps [-0.5, 0.5] across the image, so the ray origin
        // sweeps across the full view rectangle.
        rayOrigin = camera.position
            + camera.horizontal * sensorOffset.x
            + camera.vertical * sensorOffset.y;
        rayDirection = normalize(camera.direction);
    }
};

#endif // ORTHOGRAPHIC_CAMERA_SLANG
