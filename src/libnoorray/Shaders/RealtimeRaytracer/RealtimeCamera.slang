#pragma once

#include "RealtimeArgs.slang"
#include "../Raytracer/CameraRay.slang"

// The primary ray through a render pixel exactly as the primary passes trace
// it: through the pixel's jittered sample position and the aperture centre.
// With the pixel's view Z it gives the surface position, which is therefore
// not stored.
CameraRay realtimeCameraRay(uint2 pixel)
{
    RealtimeArgs* args = realtimeArgs();
    const float2 uv = (float2(pixel) + 0.5 + args.view.jitter)
        / float2(args.frame.width, args.frame.height);
    return generateCameraRay(args.frame, uv, float2(0.5), 550.0);
}

// The point of `ray` at `viewZ`, measured from the camera position along its
// forward axis.
float3 realtimeRayPoint(CameraRay ray, float viewZ)
{
    RealtimeView* view = &realtimeArgs()->view;
    const float start = dot(ray.origin - view->cameraPosition, view->cameraForward);
    return ray.origin + ray.direction
        * ((viewZ - start) / dot(ray.direction, view->cameraForward));
}
