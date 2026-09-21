#pragma once

#include "CameraLens.slang"

// Camera-ray construction shared by every ray-generation shader, so the
// spectral reference and the RGB preview frame the scene identically.

float3 cameraTransformPoint(Camera camera, float3 p)
{
    return float3(
        dot(float4(p, 1.0), float4(camera.cameraToWorld[0],
            camera.cameraToWorld[1], camera.cameraToWorld[2],
            camera.cameraToWorld[3])),
        dot(float4(p, 1.0), float4(camera.cameraToWorld[4],
            camera.cameraToWorld[5], camera.cameraToWorld[6],
            camera.cameraToWorld[7])),
        dot(float4(p, 1.0), float4(camera.cameraToWorld[8],
            camera.cameraToWorld[9], camera.cameraToWorld[10],
            camera.cameraToWorld[11])));
}

float3 cameraTransformVector(Camera camera, float3 v)
{
    return normalize(float3(
        dot(v, float3(camera.cameraToWorld[0], camera.cameraToWorld[1],
            camera.cameraToWorld[2])),
        dot(v, float3(camera.cameraToWorld[4], camera.cameraToWorld[5],
            camera.cameraToWorld[6])),
        dot(v, float3(camera.cameraToWorld[8], camera.cameraToWorld[9],
            camera.cameraToWorld[10]))));
}

struct CameraRay
{
    float3 origin;
    float3 direction;
    float weight;
    bool valid;
    // Set when the ray was refracted through the sequential lens at one
    // wavelength; a spectral integrator must then drop the others.
    bool lensTraced;
};

// `uv` is the film position in [0, 1]^2. `lensSample` in [0, 1)^2 picks the
// aperture point; 0.5 is the aperture centre.
CameraRay generateCameraRay(Frame frame, float2 uv, float2 lensSample,
    float wavelengthNm)
{
    const Camera camera = frame.camera;
    CameraRay result;
    result.valid = true;
    result.weight = 1.0;
    result.lensTraced = false;

    const float sensorWidth = max(camera.sensorWidthMm, 0.001);
    const float sensorHeight = max(camera.sensorHeightMm, 0.001);
    const float focalLength = max(camera.focalLengthMm, 0.001);
    const uint projection = camera.projection;
    const float focusDistance = max(camera.focusDistanceCm, 0.1) * 0.01;
    const float apertureRadius = max(camera.apertureDiameterMm, 0.0) * 0.0005;
    float2 normalizedSensor = (uv - 0.5) * 2.0;
    // Presentation flips the storage image back to NoorRay's historical
    // bottom-left GPU convention. Upper-left therefore already has positive
    // film Y at the displayed top; lower-left reverses that camera mapping.
    if (camera.sensorOrigin != 0u)
        normalizedSensor.y = -normalizedSensor.y;
    float3 localOrigin = float3(0.0);
    float3 localDirection;
    if (projection == 1u) // Orthographic
    {
        const float height = 10.0 * 21.0 / focalLength;
        localOrigin = float3(normalizedSensor.x * height
            * sensorWidth / sensorHeight * 0.5,
            normalizedSensor.y * height * 0.5, 0.0);
        localDirection = float3(0.0, 0.0, -1.0);
    }
    else if (projection == 2u) // Fisheye
    {
        const float aspect = sensorWidth / sensorHeight;
        const float2 p = float2(normalizedSensor.x * aspect,
            normalizedSensor.y);
        const float radius = length(p);
        if (radius < 1.0e-6)
            localDirection = float3(0.0, 0.0, -1.0);
        else
        {
            const float radiusMax = sqrt(aspect * aspect + 1.0);
            const float halfFov = atan(sensorWidth / (2.0 * focalLength));
            const float theta = radius / radiusMax * halfFov;
            localDirection = float3(p * (sin(theta) / radius), -cos(theta));
        }
    }
    else // Perspective and thin lens
    {
        localDirection = normalize(float3(
            normalizedSensor.x * sensorWidth / (2.0 * focalLength),
            normalizedSensor.y * sensorHeight / (2.0 * focalLength), -1.0));
    }
    if (projection == 4u) // Realistic sequential lens
    {
        Lens* lens = (Lens*)frame.lens;
        const uint surfaceCount = lens->surfaceCount;
        const float pupilZ = lens->rearPupilZ;
        const float pupilRadius = lens->rearPupilRadius;
        if (surfaceCount == 0u || pupilRadius <= 0.0)
            result.valid = false;
        else
        {
            const float2 square = lensSample * 2.0 - 1.0;
            float pupilR;
            float pupilPhi;
            if (abs(square.x) > abs(square.y))
            {
                pupilR = square.x;
                pupilPhi = 0.785398163 * square.y
                    / (square.x == 0.0 ? 1.0 : square.x);
            }
            else
            {
                pupilR = square.y;
                pupilPhi = 1.570796327 - 0.785398163 * square.x
                    / (square.y == 0.0 ? 1.0 : square.y);
            }
            const float2 pupil = pupilRadius * pupilR
                * float2(cos(pupilPhi), sin(pupilPhi));
            float3 lensOrigin = float3(-normalizedSensor.x * sensorWidth * 0.5,
                -normalizedSensor.y * sensorHeight * 0.5, 0.0);
            float3 lensDirection = normalize(float3(pupil, pupilZ) - lensOrigin);
            result.lensTraced = true;
            result.valid = traceLensFilmToWorld(lens, lensOrigin,
                lensDirection, wavelengthNm);
            if (result.valid)
            {
                const float cosineSquared = lensDirection.z * lensDirection.z;
                result.weight = 3.141592654 * pupilRadius * pupilRadius
                    * cosineSquared * cosineSquared
                    / max(1.0e-6, pupilZ * pupilZ);
                localOrigin = lensOrigin * 0.001;
                localDirection = normalize(float3(lensDirection.xy,
                    -lensDirection.z));
            }
        }
    }
    else if ((projection == 2u || projection == 3u)
        && apertureRadius > 0.0)
    {
        const float angle = 6.28318530718 * lensSample.x;
        const float radius = apertureRadius * sqrt(lensSample.y);
        localOrigin = float3(cos(angle) * radius, sin(angle) * radius, 0.0);
        const float3 focusPoint = localDirection
            * (focusDistance / max(-localDirection.z, 1.0e-5));
        localDirection = normalize(focusPoint - localOrigin);
    }

    result.origin = cameraTransformPoint(camera, localOrigin);
    result.direction = cameraTransformVector(camera, localDirection);
    return result;
}
