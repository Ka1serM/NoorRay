#ifndef ICAMERA_SLANG
#define ICAMERA_SLANG

#include "../Shared.h"
#include "../Bindings.slang"
#include "../Samplers/ISampler.slang"

// Maps a pixel coordinate to a perspective ray via the CameraData image-plane
// parameterisation. Shared by PerspectiveCamera and ThinLensCamera.
void perspectiveBaseRay(
    int2 pixelCoord,
    int2 screenSize,
    float2 jitter,
    out float3 rayOrigin,
    out float3 rayDirection)
{
    CameraData camera = pushConstants.camera;
    float2 uv = (float2(pixelCoord) + jitter) / float2(screenSize);
    float2 sensorOffset = float2(uv.x - 0.5, 0.5 - uv.y);

    float3 imagePlanePoint = camera.position
        + camera.direction  * camera.focalLength
        + camera.horizontal * camera.focalLength * sensorOffset.x
        + camera.vertical   * camera.focalLength * sensorOffset.y;

    rayOrigin    = camera.position;
    rayDirection = normalize(imagePlanePoint - rayOrigin);
}

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
