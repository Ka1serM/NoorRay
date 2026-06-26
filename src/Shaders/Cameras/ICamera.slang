#ifndef ICAMERA_SLANG
#define ICAMERA_SLANG

#include "../Shared.h"
#include "../Samplers/ISampler.slang"

struct LocalRay
{
    float3 origin;
    float3 direction;
};

#ifndef RAY_LUT_GENERATION_SHADER

// Maps a pixel coordinate to a perspective ray via the CameraData image-plane
// parameterisation. Shared by PerspectiveCamera and ThinLensCamera.
void perspectiveBaseRay(
    int2 pixelCoord,
    int2 screenSize,
    float2 jitter,
    out float3 rayOrigin,
    out float3 rayDirection)
{
    CameraData camera = sceneSettings.camera;
    float2 uv = (float2(pixelCoord) + jitter) / float2(screenSize);
    float2 sensorOffset = float2(uv.x - 0.5, 0.5 - uv.y);

    float3 right   = pushConstants.cameraToWorld[0].xyz;
    float3 up      = pushConstants.cameraToWorld[1].xyz;
    float3 forward = pushConstants.cameraToWorld[2].xyz;
    float3 pos     = pushConstants.cameraToWorld[3].xyz;

    float3 imagePlanePoint = pos
        + forward * camera.focalLength
        + right   * camera.focalLength * camera.sensorScaleX * sensorOffset.x
        + up      * camera.focalLength * camera.sensorScaleY * sensorOffset.y;

    rayOrigin    = pos;
    rayDirection = normalize(imagePlanePoint - pos);
}

void transformCameraLocalRayToWorld(LocalRay localRay, out float3 rayOrigin, out float3 rayDirection)
{
    float3 right   = pushConstants.cameraToWorld[0].xyz;
    float3 up      = pushConstants.cameraToWorld[1].xyz;
    float3 forward = pushConstants.cameraToWorld[2].xyz;
    float3 pos     = pushConstants.cameraToWorld[3].xyz;
    rayOrigin = pos
        + right   * localRay.origin.x
        + up      * localRay.origin.y
        + forward * localRay.origin.z;
    rayDirection = normalize(
        right   * localRay.direction.x
        + up      * localRay.direction.y
        + forward * localRay.direction.z);
}

#endif // RAY_LUT_GENERATION_SHADER

// Concentric disk sample with a radial power bias for bokeh shape control.
// edgeBias > 1 → rounder disk, < 1 → softer edge.
float2 bokehDisk(float2 u, float edgeBias)
{
    float2 uv = 2.0 * u - 1.0;
    if (abs(uv.x) <= 1e-6 && abs(uv.y) <= 1e-6)
        return float2(0.0);

    float r, theta;
    if (abs(uv.x) > abs(uv.y)) {
        r     = uv.x;
        theta = 0.25 * 3.14159265 * (uv.y / uv.x);
    } else {
        r     = uv.y;
        theta = 0.5 * 3.14159265 - 0.25 * 3.14159265 * (uv.x / uv.y);
    }
    float2 d   = r * float2(cos(theta), sin(theta));
    float  len = length(d);
    if (len <= 1e-6) return float2(0.0);
    float newLen = pow(len, 1.0 / max(edgeBias, 1e-4));
    return d * (newLen / len);
}

interface ICamera<TSampler>
    where TSampler : ISampler
{
    void generateRay(
        int2 pixelCoord,
        int2 screenSize,
        inout SamplerState samplerState,
        inout TSampler sampler,
        bool deterministicSample,
        out float3 rayOrigin,
        out float3 rayDirection);
};

#endif // ICAMERA_SLANG
