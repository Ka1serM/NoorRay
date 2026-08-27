#pragma once

// Fixed-size screen-space gizmo for a scene object (currently lights). Mirrors
// ViewportBillboard in Viewport.h; kept vec4-only so its layout can't drift from
// the C++ struct regardless of packing convention.
struct ViewportBillboard
{
    float4 positionType; // xyz = world position, w = object type (LightInstance::Type*)
    float4 color;        // rgb = display color
};

struct PushConstants
{
    float4x4 viewProjection;
    float2   screenSize;
    float    radius; // billboard half-size, in pixels
    uint     billboards;
};

struct BillboardRoot
{
    PushConstants* arguments;
};

[[vk::push_constant]] ConstantBuffer<BillboardRoot> root;

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
};

[shader("vertex")]
VertexOutput vertMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    const PushConstants pushConstants = *root.arguments;
    StructuredBuffer<ViewportBillboard> billboardBuffer =
        DescriptorHandle<StructuredBuffer<ViewportBillboard>>(uint2(pushConstants.billboards, 0));
    const ViewportBillboard billboard = billboardBuffer[instanceId];
    const float4 clip = mul(pushConstants.viewProjection, float4(billboard.positionType.xyz, 1.0));

    VertexOutput output;
    output.color = billboard.color.rgb;
    output.type = int(billboard.positionType.w);

    if (clip.w <= 0.0)
    {
        // Behind the camera — collapse to a degenerate (zero-area) triangle so
        // nothing gets rasterized, rather than branching in the fragment stage.
        output.clipPosition = float4(0.0, 0.0, 0.0, 1.0);
        output.local = float2(0.0, 0.0);
        return output;
    }

    const float2 corner = corners[vertexId];
    const float2 ndc = clip.xy / clip.w;
    // The raw output is presented with a single V flip, so use its bottom-left
    // convention here as well. Scaling by clip.w preserves the pixel size
    // through the perspective divide.
    const float2 ndcOffset = corner * pushConstants.radius * 2.0 / pushConstants.screenSize;

    output.clipPosition = float4((ndc + ndcOffset) * clip.w, clip.z, clip.w);
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
    const PushConstants pushConstants = *root.arguments;
    // Antialias in local billboard units. The glyphs are strokes, not filled SDF
    // blobs, which keeps them crisp and avoids a heavy outline.
    const float aa = 1.15 / pushConstants.radius;
    const float alpha = billboardAlpha(input.local, input.type, aa);
    if (alpha <= 0.003)
        discard;

    return float4(input.color, alpha);
}
