#pragma once

// Wächter and Binder, "A Fast and Robust Method for Avoiding Self-Intersection",
// Ray Tracing Gems chapter 6. The offset is whole float steps, so it follows
// the position's magnitude instead of a fixed world-space distance.

static const float RayOriginNearZero = 1.0 / 32.0;
static const float RayOriginFloatScale = 1.0 / 65536.0;
static const float RayOriginIntScale = 256.0;

// Moves `position` off the surface to the side of `direction`.
float3 offsetRayOrigin(float3 position, float3 geometricNormal, float3 direction)
{
    const float3 normal = dot(geometricNormal, direction) >= 0.0
        ? geometricNormal : -geometricNormal;
    const int3 offset = int3(RayOriginIntScale * normal);
    const float3 steppedPosition = asfloat(asint(position)
        + select(position < 0.0, -offset, offset));
    return select(abs(position) < RayOriginNearZero,
        position + RayOriginFloatScale * normal, steppedPosition);
}
