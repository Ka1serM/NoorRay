#pragma once

#include "../../Shared/Viewport.h"

[[vk::push_constant]] ConstantBuffer<ViewportBillboardRoot> root;

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
    nointerpolation uint icon : ICON;
    nointerpolation float distanceFade : DISTANCEFADE;
};

// Each instance is a fixed-size plane at the object's world position. The
// projection's first two rows give camera right/up, including camera roll.
float4 billboardCorner(ViewportBillboardPushConstants args, float3 center, float2 corner)
{
    const float3 right = normalize(args.viewProjection[0].xyz);
    const float3 up = normalize(args.viewProjection[1].xyz);
    const float3 world = center + (right * corner.x + up * corner.y) * args.worldHalfSize;
    return mul(args.viewProjection, float4(world, 1.0));
}

// The icons are an editing overlay and intentionally render through level
// geometry. Fade them only by their distance to the camera: this prevents a
// large glyph from clipping across the viewport when the camera enters it,
// and keeps distant map-wide lights out of the editing view.
float billboardDistanceFade(ViewportBillboardPushConstants args, float3 center)
{
    if (args.cameraPosition.w == 0.0)
        return 1.0;
    const float distanceToCamera = length(center - args.cameraPosition.xyz);
    return smoothstep(args.distanceFade.x, args.distanceFade.y, distanceToCamera)
        * (1.0 - smoothstep(args.distanceFade.z, args.distanceFade.w, distanceToCamera));
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
    output.icon = uint(billboard.positionType.w);
    output.distanceFade = billboardDistanceFade(pushConstants, billboard.positionType.xyz);

    if (clip.w <= 0.0 || billboard.color.a == 0.0)
    {
        // Behind the camera or hidden — collapse to a degenerate (zero-area) triangle so
        // nothing gets rasterized, rather than branching in the fragment stage.
        output.clipPosition = float4(0.0, 0.0, 0.0, 1.0);
        output.local = float2(0.0, 0.0);
        return output;
    }

    const float2 corner = corners[vertexId];
    const float4 planeClip = billboardCorner(pushConstants, billboard.positionType.xyz, corner);
    // Map the projected plane into the logical rectangle of the attachment.
    const float2 targetClip = (planeClip.xy + planeClip.w) * pushConstants.targetScale - planeClip.w;
    output.clipPosition = float4(targetClip, planeClip.z, planeClip.w);
    output.local = corner;
    return output;
}

float luminance(const float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float msdfMedian(const float3 value)
{
    return max(min(value.r, value.g), min(max(value.r, value.g), value.b));
}

// Coverage of the icon's multi-channel distance texture at local (-1..1 over
// the plane, y up). screenPxRange is how many screen pixels the texture's
// distance range spans, which sets the antialiasing width.
float billboardAlpha(const ViewportBillboardPushConstants args, const float2 local,
    const uint icon, const float screenPxRange)
{
    const uint textureHandle = args.iconTextures[icon];
    Texture2D<float4> image = ResourceDescriptorHeap[textureHandle];
    SamplerState sampler = SamplerDescriptorHeap[args.iconSampler];
    // Texture rows run top to bottom.
    const float2 uv = float2(local.x * 0.5 + 0.5, 0.5 - local.y * 0.5);
    const float distance = msdfMedian(image.SampleLevel(sampler, uv, 0.0).rgb) - 0.5;
    return saturate(distance * max(screenPxRange, 1.0) + 0.5);
}

[shader("fragment")]
float4 fragMain(VertexOutput input) : SV_Target
{
    const ViewportBillboardPushConstants pushConstants = *root.arguments;
    const float2 unitsPerPixel = fwidth(input.local) * 0.5;
    const float screenPxRange = 0.5 * (ViewportBillboardDistanceRange / ViewportBillboardTextureSize)
        * (1.0 / max(unitsPerPixel.x, 1.0e-5) + 1.0 / max(unitsPerPixel.y, 1.0e-5));
    const float alpha = billboardAlpha(pushConstants, input.local, input.icon, screenPxRange)
        * input.distanceFade;
    if (alpha <= 0.003)
        discard;

    return float4(input.color, alpha);
}


// Light picking. A separate buffer holds one uint per output pixel: 0 where no
// icon is, otherwise the billboard index + 1. Clear resets the logical
// rectangle each frame; Stamp then writes each visible glyph, one group
// per billboard. Overlaps keep the highest index - the icon drawn last, which
// is the one blended on top.
static const uint LightIdGroupSize = 16u;

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
    const float distanceFade = billboardDistanceFade(pushConstants, billboard.positionType.xyz);
    if (clip.w <= 0.0 || distanceFade <= 0.003)
        return;

    // Same bottom-left pixel space the vertex stage draws the icon into.
    const float2 center = (clip.xy / clip.w * 0.5 + 0.5) * pushConstants.screenSize;
    const float4 edge = billboardCorner(pushConstants, billboard.positionType.xyz, float2(1.0));
    const float2 edgePixel = (edge.xy / edge.w * 0.5 + 0.5) * pushConstants.screenSize;
    const float2 halfExtent = abs(edgePixel - center);
    if (any(halfExtent <= 0.0)) return;
    const int2 size = int2(pushConstants.screenSize);
    const int2 origin = clamp(int2(floor(center - halfExtent)), int2(0), size);
    const int2 end = clamp(int2(ceil(center + halfExtent)), int2(0), size);
    const int2 extent = end - origin;
    uint* lightIds = pushConstants.lightIds;
    uint previous;

    for (int y = int(threadId.y); y < extent.y; y += int(LightIdGroupSize))
    {
        for (int x = int(threadId.x); x < extent.x; x += int(LightIdGroupSize))
        {
            const int2 pixel = origin + int2(x, y);
            if (any(pixel < 0) || any(pixel >= size))
                continue;
            const float2 delta = float2(pixel) + 0.5 - center;
            const float2 local = delta / halfExtent;
            const float screenPxRange = ViewportBillboardDistanceRange / ViewportBillboardTextureSize
                * 2.0 * min(halfExtent.x, halfExtent.y);
            if (billboardAlpha(pushConstants, local, uint(billboard.positionType.w), screenPxRange) <= 0.05)
                continue;
            InterlockedMax(lightIds[uint(pixel.y) * pushConstants.lightIdStride + uint(pixel.x)],
                index + 1u, previous);
        }
    }
}
