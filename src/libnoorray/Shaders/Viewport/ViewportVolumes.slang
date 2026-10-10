#pragma once

import noorrhi;

#include "../../Shared/Viewport.h"

[[vk::push_constant]] ConstantBuffer<ViewportVolumeRoot> root;

// x is 0 at a segment's first end point and 1 at its second; y is the side.
static const float2 corners[6] = {
    float2(0, -1), float2(1, -1), float2(0,  1),
    float2(0,  1), float2(1, -1), float2(1,  1)
};

// Clip-space w below this is behind the camera. The segment is cut there so
// the perspective divide never flips; the rasterizer clips the rest.
static const float MinimumW = 1.0e-3;

struct VertexOutput
{
    float4 clipPosition : SV_Position;
    // Signed pixel distance from the line's centre.
    float across : ACROSS;
    nointerpolation float halfWidth : HALFWIDTH;
    nointerpolation float4 color : COLOR;
    nointerpolation uint volumeIndex : VOLUMEINDEX;
    float viewZ : VIEWZ;
};

[shader("vertex")]
VertexOutput vertMain(uint vertexId : SV_VertexID, uint segment : SV_InstanceID)
{
    const ViewportVolumePushConstants pushConstants = *root.arguments;
    float3* points = pushConstants.segments;
    float4 first = mul(pushConstants.modelViewProjection, float4(points[2 * segment], 1.0));
    float4 second = mul(pushConstants.modelViewProjection, float4(points[2 * segment + 1], 1.0));

    VertexOutput output;
    output.across = 0.0;
    output.halfWidth = 0.0;
    output.color = float4(0.0);
    output.volumeIndex = pushConstants.volumeIndex;
    output.viewZ = 0.0;
    output.clipPosition = float4(0.0, 0.0, 0.0, 1.0);
    if (first.w < MinimumW && second.w < MinimumW)
        return output;
    if (first.w < MinimumW)
        first = lerp(first, second, (MinimumW - first.w) / (second.w - first.w));
    else if (second.w < MinimumW)
        second = lerp(second, first, (MinimumW - second.w) / (first.w - second.w));

    const float2 firstPixel = first.xy / first.w * 0.5 * pushConstants.screenSize;
    const float2 secondPixel = second.xy / second.w * 0.5 * pushConstants.screenSize;
    const float2 delta = secondPixel - firstPixel;
    const float2 direction = dot(delta, delta) > 1.0e-8 ? normalize(delta) : float2(1.0, 0.0);
    const float2 normal = float2(-direction.y, direction.x);
    // One extra pixel carries the antialiased fringe.
    const float halfWidth = pushConstants.lineWidth * 0.5 + 1.0;

    const float2 corner = corners[vertexId];
    const float4 end = corner.x == 0.0 ? first : second;
    const float2 offsetPixel = normal * (corner.y * halfWidth)
        + direction * ((corner.x * 2.0 - 1.0) * halfWidth);
    const float2 clipXY = end.xy + offsetPixel / pushConstants.screenSize * 2.0 * end.w;
    // Map into the logical rectangle of the attachment, as the billboards do.
    const float2 targetClip = (clipXY + end.w) * pushConstants.targetScale - end.w;
    output.clipPosition = float4(targetClip, end.z, end.w);
    output.across = corner.y * halfWidth;
    output.halfWidth = pushConstants.lineWidth * 0.5;
    output.color = pushConstants.color;
    output.volumeIndex = pushConstants.volumeIndex;
    output.viewZ = end.w;
    return output;
}

// A depth sample hides the fragment when its surface is nearer than the
// fragment's (device depth is nearPlane / viewZ, larger when nearer), with a
// small relative margin so a line lying on a surface is not at the mercy of
// the depth image's per-frame jitter.
static const float DepthMargin = 1.0e-3;

float depthVisible(float surfaceDepth, float fragmentDepth)
{
    return surfaceDepth <= fragmentDepth * (1.0 + DepthMargin) + 1.0e-6 ? 1.0 : 0.0;
}

// The share of the fragment the scene leaves visible. The depth image is
// allocated at output size but only its origin rectangle, traceSize pixels,
// holds the render. Render pixel p sampled the scene at p + 0.5 + jitter in
// render pixels, and the fragment sits at logical pixel centre pos, so the
// continuous render coordinate is pos * traceSize / screenSize - 0.5 - jitter.
// Each of the four surrounding samples is tested on its own and the results
// blend bilinearly: blending the depths themselves would invent surfaces
// between a silhouette and what lies behind it, and a hard test flips with
// every jitter. The raytracer stores nearPlane / viewZ (1 cm near plane), so
// the test compares camera-space Z rather than raster depth conventions.
float sceneVisibility(ViewportVolumePushConstants args, float viewZ, float2 position)
{
    const float2 source = position * args.traceSize / args.screenSize - 0.5 - args.depthJitter;
    const float2 base = floor(source);
    const float2 fraction = source - base;
    const float2 lastPixel = args.traceSize - 1.0;
    const int2 low = int2(clamp(base, float2(0.0), lastPixel));
    const int2 high = int2(clamp(base + 1.0, float2(0.0), lastPixel));
    RWTexture2D<float4> depth = ResourceDescriptorHeap[args.depthImage];
    const float fragmentDepth = 1.0 / max(viewZ, 1.0);
    const float top = lerp(depthVisible(depth[low].x, fragmentDepth),
        depthVisible(depth[int2(high.x, low.y)].x, fragmentDepth), fraction.x);
    const float bottom = lerp(depthVisible(depth[int2(low.x, high.y)].x, fragmentDepth),
        depthVisible(depth[high].x, fragmentDepth), fraction.x);
    return lerp(top, bottom, fraction.y);
}

[shader("fragment")]
float4 fragMain(VertexOutput input) : SV_Target
{
    const ViewportVolumePushConstants args = *root.arguments;
    float visibility = 1.0;
    if (args.depthImage != 0u)
    {
        visibility = sceneVisibility(args, input.viewZ, input.clipPosition.xy);
        if (visibility <= 0.0)
            discard;
    }
    // The quad includes one pixel of antialias fringe on either side. Use it
    // as a slightly wider click target while the visible line stays thin.
    if (abs(input.across) >= input.halfWidth + 1.0)
        discard;
    const uint2 outputPixel = uint2(input.clipPosition.xy);
    if (visibility >= 0.5 && all(outputPixel < uint2(args.screenSize * args.targetScale)))
    {
        uint* ids = args.volumeIds;
        uint previous;
        InterlockedMax(ids[outputPixel.y * args.idStride + outputPixel.x],
            input.volumeIndex + 1u, previous);
    }
    const float coverage = saturate(input.halfWidth + 0.5 - abs(input.across));
    const float alpha = input.color.a * coverage * visibility;
    return float4(input.color.rgb, alpha);
}
