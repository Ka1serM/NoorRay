#pragma once

#include "../../Shared/Viewport.h"

[[vk::push_constant]] ConstantBuffer<ViewportBillboardRoot> root;

static const int BillboardTypePoint       = 0;
static const int BillboardTypeSpot        = 1;
static const int BillboardTypeRect        = 2;
static const int BillboardTypeDirectional = 3;

// The clean graphics API currently draws triangle lists. Emit two explicit
// triangles; four strip-ordered vertices only rasterized the first half.
static const float2 corners[6] = {
    float2(-1, -1), float2( 1, -1), float2(-1,  1),
    float2(-1,  1), float2( 1, -1), float2( 1,  1)
};

struct VertexOutput
{
    float4 clipPosition : SV_Position;
    float2 local         : LOCAL;
    float3 color          : COLOR;
    nointerpolation int type : TYPE;
    nointerpolation float radius : RADIUS;
};

// Icon half-size in pixels for a billboard whose clip-space w (its view-space
// distance under a perspective camera) is viewDistance.
float billboardRadius(ViewportBillboardPushConstants pushConstants, float viewDistance)
{
    const float range = max(pushConstants.farDistance - pushConstants.nearDistance, 1.0e-4);
    const float t = saturate((viewDistance - pushConstants.nearDistance) / range);
    return lerp(pushConstants.maxRadius, pushConstants.minRadius, t);
}

[shader("vertex")]
VertexOutput vertMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    const ViewportBillboardPushConstants pushConstants = *root.arguments;
    ViewportBillboard* billboardBuffer = pushConstants.billboards;
    const ViewportBillboard billboard = billboardBuffer[instanceId];
    const float4 clip = mul(pushConstants.viewProjection, float4(billboard.positionType.xyz, 1.0));

    VertexOutput output;
    output.color = instanceId == pushConstants.selectedBillboard
        ? ViewportSelectionColor : billboard.color.rgb;
    output.type = int(billboard.positionType.w);
    output.radius = billboardRadius(pushConstants, clip.w) * pushConstants.radiusScale;

    if (clip.w <= 0.0)
    {
        // Behind the camera — collapse to a degenerate (zero-area) triangle so
        // nothing gets rasterized, rather than branching in the fragment stage.
        output.clipPosition = float4(0.0, 0.0, 0.0, 1.0);
        output.local = float2(0.0, 0.0);
        output.radius = 0.0;
            return output;
    }

    const float2 corner = corners[vertexId];
    const float2 ndc = clip.xy / clip.w;
    // The raw output is presented with a single V flip, so use its bottom-left
    // convention here as well. Scaling by clip.w preserves the pixel size
    // through the perspective divide.
    const float2 ndcOffset = corner * output.radius * 2.0 / pushConstants.screenSize;
    // The attachment may be larger than the render; squeeze the logical NDC
    // range into its bottom-left rectangle, which is what gets presented.
    const float2 targetNdc = (ndc + ndcOffset + 1.0) * pushConstants.targetScale - 1.0;

    output.clipPosition = float4(targetNdc * clip.w, clip.z, clip.w);
    output.local = corner;
    return output;
}

float luminance(const float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float sdCircle(const float2 p, const float r)
{
    return length(p) - r;
}

float sdBox(const float2 p, const float2 b)
{
    const float2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float sdSegment(const float2 p, const float2 a, const float2 b)
{
    const float2 ab = b - a;
    const float h = saturate(dot(p - a, ab) / dot(ab, ab));
    return length(p - (a + ab * h));
}

float stroke(const float distanceToCenterline, const float width, const float aa)
{
    return 1.0 - smoothstep(width - aa, width + aa, distanceToCenterline);
}

float ringStroke(const float2 p, const float radius, const float width, const float aa)
{
    return stroke(abs(sdCircle(p, radius)), width, aa);
}

float boxStroke(const float2 p, const float2 halfSize, const float width, const float aa)
{
    return stroke(abs(sdBox(p, halfSize)), width, aa);
}

float segmentStroke(const float2 p, const float2 a, const float2 b, const float width, const float aa)
{
    return stroke(sdSegment(p, a, b), width, aa);
}

float pointLightAlpha(const float2 p, const float aa)
{
    const float w = 0.045;
    float a = ringStroke(p, 0.34, w, aa);
    a = max(a, stroke(length(p), 0.055, aa));
    a = max(a, segmentStroke(p, float2(-0.68,  0.00), float2(-0.49,  0.00), w, aa));
    a = max(a, segmentStroke(p, float2( 0.49,  0.00), float2( 0.68,  0.00), w, aa));
    a = max(a, segmentStroke(p, float2( 0.00, -0.68), float2( 0.00, -0.49), w, aa));
    a = max(a, segmentStroke(p, float2( 0.00,  0.49), float2( 0.00,  0.68), w, aa));
    a = max(a, segmentStroke(p, float2(-0.48, -0.48), float2(-0.35, -0.35), w, aa));
    a = max(a, segmentStroke(p, float2( 0.35, -0.35), float2( 0.48, -0.48), w, aa));
    a = max(a, segmentStroke(p, float2(-0.48,  0.48), float2(-0.35,  0.35), w, aa));
    a = max(a, segmentStroke(p, float2( 0.35,  0.35), float2( 0.48,  0.48), w, aa));
    return a;
}

float spotLightAlpha(const float2 p, const float aa)
{
    const float w = 0.045;
    float a = ringStroke(p - float2(0.0, -0.55), 0.12, w, aa);
    a = max(a, segmentStroke(p, float2(-0.06, -0.42), float2(-0.65, 0.50), w, aa));
    a = max(a, segmentStroke(p, float2( 0.06, -0.42), float2( 0.65, 0.50), w, aa));
    a = max(a, segmentStroke(p, float2(-0.65, 0.50), float2( 0.65, 0.50), w, aa));
    return a;
}

float rectLightAlpha(const float2 p, const float aa)
{
    const float w = 0.045;
    float a = boxStroke(p, float2(0.56, 0.40), w, aa);
    a = max(a, segmentStroke(p, float2(-0.36, -0.58), float2( 0.36, -0.58), w, aa));
    a = max(a, segmentStroke(p, float2(-0.56, -0.40), float2(-0.36, -0.58), w, aa));
    a = max(a, segmentStroke(p, float2( 0.56, -0.40), float2( 0.36, -0.58), w, aa));
    a = max(a, segmentStroke(p, float2(-0.36,  0.58), float2( 0.36,  0.58), w, aa));
    a = max(a, segmentStroke(p, float2(-0.56,  0.40), float2(-0.36,  0.58), w, aa));
    a = max(a, segmentStroke(p, float2( 0.56,  0.40), float2( 0.36,  0.58), w, aa));
    return a;
}

float directionalLightAlpha(const float2 p, const float aa)
{
    const float w = 0.045;
    float a = ringStroke(p, 0.28, w, aa);
    a = max(a, segmentStroke(p, float2(-0.58,  0.00), float2(-0.42,  0.00), w, aa));
    a = max(a, segmentStroke(p, float2( 0.42,  0.00), float2( 0.58,  0.00), w, aa));
    a = max(a, segmentStroke(p, float2( 0.00, -0.58), float2( 0.00, -0.42), w, aa));
    a = max(a, segmentStroke(p, float2( 0.00,  0.42), float2( 0.00,  0.58), w, aa));
    a = max(a, segmentStroke(p, float2(-0.41, -0.41), float2(-0.30, -0.30), w, aa));
    a = max(a, segmentStroke(p, float2( 0.30, -0.30), float2( 0.41, -0.41), w, aa));
    a = max(a, segmentStroke(p, float2(-0.41,  0.41), float2(-0.30,  0.30), w, aa));
    a = max(a, segmentStroke(p, float2( 0.30,  0.30), float2( 0.41,  0.41), w, aa));
    a = max(a, segmentStroke(p, float2( 0.45, -0.62), float2( 0.71, -0.36), w, aa));
    a = max(a, segmentStroke(p, float2( 0.71, -0.36), float2( 0.62, -0.34), w, aa));
    a = max(a, segmentStroke(p, float2( 0.71, -0.36), float2( 0.69, -0.45), w, aa));
    return a;
}

float billboardAlpha(const float2 p, const int type, const float aa)
{
    switch (type)
    {
    case BillboardTypePoint:
        return pointLightAlpha(p, aa);
    case BillboardTypeSpot:
        return spotLightAlpha(p, aa);
    case BillboardTypeRect:
        return rectLightAlpha(p, aa);
    default:
        return directionalLightAlpha(p, aa);
    }
}

[shader("fragment")]
float4 fragMain(VertexOutput input) : SV_Target
{
    // Antialias in local billboard units. The glyphs are strokes, not filled SDF
    // blobs, which keeps them crisp and avoids a heavy outline.
    const float aa = 1.15 / input.radius;
    const float alpha = billboardAlpha(input.local, input.type, aa);
    if (alpha <= 0.003)
        discard;

    return float4(input.color, alpha);
}


// Light picking. A separate buffer holds one uint per output pixel: 0 where no
// icon is, otherwise the billboard index + 1. Clear resets the logical
// rectangle each frame; Stamp then writes a filled circle per light, one group
// per billboard. Overlaps keep the highest index - the icon drawn last, which
// is the one blended on top.
static const uint LightIdGroupSize = 16u;
static const int LightIdMaxExtent = 64;

[shader("compute")]
[numthreads(LightIdGroupSize, LightIdGroupSize, 1)]
void lightIdClear(uint3 threadId : SV_DispatchThreadID)
{
    const ViewportBillboardPushConstants pushConstants = *root.arguments;
    if (any(float2(threadId.xy) >= pushConstants.screenSize))
        return;
    uint* lightIds = pushConstants.lightIds;
    lightIds[threadId.y * pushConstants.lightIdStride + threadId.x] = 0u;
}

[shader("compute")]
[numthreads(LightIdGroupSize, LightIdGroupSize, 1)]
void lightIdStamp(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID)
{
    const ViewportBillboardPushConstants pushConstants = *root.arguments;
    const uint index = groupId.x;
    if (index >= pushConstants.billboardCount)
        return;
    ViewportBillboard* billboardBuffer = pushConstants.billboards;
    const ViewportBillboard billboard = billboardBuffer[index];
    const float4 clip = mul(pushConstants.viewProjection, float4(billboard.positionType.xyz, 1.0));
    if (clip.w <= 0.0)
        return;

    // Same bottom-left pixel space the vertex stage draws the icon into.
    const float2 center = (clip.xy / clip.w * 0.5 + 0.5) * pushConstants.screenSize;
    const float radius = billboardRadius(pushConstants, clip.w) * pushConstants.radiusScale;
    const int2 origin = int2(floor(center - radius));
    const int extent = min(int(ceil(radius * 2.0)) + 1, LightIdMaxExtent);
    const int2 size = int2(pushConstants.screenSize);
    uint* lightIds = pushConstants.lightIds;
    uint previous;

    for (int y = int(threadId.y); y < extent; y += int(LightIdGroupSize))
    {
        for (int x = int(threadId.x); x < extent; x += int(LightIdGroupSize))
        {
            const int2 pixel = origin + int2(x, y);
            if (any(pixel < 0) || any(pixel >= size))
                continue;
            const float2 delta = float2(pixel) + 0.5 - center;
            if (dot(delta, delta) > radius * radius)
                continue;
            InterlockedMax(lightIds[uint(pixel.y) * pushConstants.lightIdStride + uint(pixel.x)],
                index + 1u, previous);
        }
    }
}
