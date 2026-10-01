#pragma once

// Ray differentials (Igehy, "Tracing Ray Differentials", 1999): how a path
// vertex and the ray through it move when the camera ray moves one render
// pixel along the image's x or y. Transferred to a hit they give the
// screen-space derivatives of everything interpolated over its triangle,
// which materials read as ddx/ddy and texture filtering as the footprint.
//
// Bounces are followed as path differentials with the random numbers held
// fixed: a sampled direction keeps its place relative to the shading frame
// (and a reflection its half vector), so the differential follows the change
// of the frame, the normal and the incoming ray across the pixel.
struct VertexDifferential
{
    // The vertex: a camera ray's origin, or where a ray hit.
    float3 positionDx;
    float3 positionDy;
    // The vertex's shading normal; zero at the camera.
    float3 normalDx;
    float3 normalDy;
    // The direction of the ray through the vertex: leaving the camera, or
    // arriving at a hit.
    float3 directionDx;
    float3 directionDy;
};

VertexDifferential zeroVertexDifferential()
{
    return (VertexDifferential)0;
}

// Igehy's transfer: the ray moved to distance t, then along itself onto the
// plane with normal `planeNormal`, so the result lies in the surface.
float3 transferToPlane(float3 originDelta, float3 directionDelta, float3 direction, float t,
    float3 planeNormal)
{
    const float3 moved = originDelta + t * directionDelta;
    const float cosine = dot(direction, planeNormal);
    const float denominator = abs(cosine) > 1.0e-6 ? cosine : (cosine < 0.0 ? -1.0e-6 : 1.0e-6);
    return moved - direction * (dot(moved, planeNormal) / denominator);
}

// The differential of the ray `ray` arriving at a surface with plane normal
// `planeNormal` at distance t. The hit's normal terms are the caller's.
VertexDifferential arriveAt(VertexDifferential ray, float3 direction, float t, float3 planeNormal)
{
    VertexDifferential hit = ray;
    hit.positionDx = transferToPlane(ray.positionDx, ray.directionDx, direction, t, planeNormal);
    hit.positionDy = transferToPlane(ray.positionDy, ray.directionDy, direction, t, planeNormal);
    hit.normalDx = float3(0.0);
    hit.normalDy = float3(0.0);
    return hit;
}

// A direction fixed in the shading frame turns with it: by the rotation that
// takes the normal to normal + normalDelta.
float3 turnWithFrame(float3 direction, float3 normal, float3 normalDelta)
{
    return normalDelta * dot(normal, direction) - normal * dot(normalDelta, direction);
}

// The reflection of `incoming` about `half` changed by the changes of both.
float3 reflectDelta(float3 incoming, float3 incomingDelta, float3 halfVector, float3 halfDelta)
{
    const float projection = dot(incoming, halfVector);
    const float projectionDelta = dot(incomingDelta, halfVector) + dot(incoming, halfDelta);
    return incomingDelta - 2.0 * (projection * halfDelta + projectionDelta * halfVector);
}

// The ray leaving `hit` along `outgoing`, having arrived along `incoming`. A
// specular sample is the reflection about a half vector fixed in the shading
// frame; any other sample is fixed in the frame itself.
VertexDifferential leave(VertexDifferential hit, float3 incoming, float3 normal, float3 outgoing,
    bool specular)
{
    VertexDifferential ray = hit;
    if (specular)
    {
        const float3 halfVector = normalize(outgoing - incoming);
        ray.directionDx = reflectDelta(incoming, hit.directionDx, halfVector,
            turnWithFrame(halfVector, normal, hit.normalDx));
        ray.directionDy = reflectDelta(incoming, hit.directionDy, halfVector,
            turnWithFrame(halfVector, normal, hit.normalDy));
    }
    else
    {
        ray.directionDx = turnWithFrame(outgoing, normal, hit.normalDx);
        ray.directionDy = turnWithFrame(outgoing, normal, hit.normalDy);
    }
    ray.normalDx = float3(0.0);
    ray.normalDy = float3(0.0);
    return ray;
}
