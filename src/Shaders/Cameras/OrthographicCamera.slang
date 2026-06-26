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
        CameraData camera = sceneSettings.camera;
        float2 uv = deterministicSample
            ? (float2(pixelCoord) + 0.5) / float2(screenSize)
            : (float2(pixelCoord) + sampler.get2D(samplerState)) / float2(screenSize);
        float2 sensorOffset = float2(uv.x - 0.5, 0.5 - uv.y);

        float3 right   = pushConstants.cameraToWorld[0].xyz;
        float3 up      = pushConstants.cameraToWorld[1].xyz;
        float3 forward = pushConstants.cameraToWorld[2].xyz;
        float3 pos     = pushConstants.cameraToWorld[3].xyz;

        rayOrigin = pos
            + right * camera.sensorScaleX * sensorOffset.x
            + up    * camera.sensorScaleY * sensorOffset.y;
        rayDirection = forward;
    }
};

#endif // ORTHOGRAPHIC_CAMERA_SLANG
