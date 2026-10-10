#pragma once

import noorrhi;

#include "../../Shared/Viewport.h"

static const int GROUP_SIZE = 16;

[[vk::push_constant]] ConstantBuffer<ViewportCompositeRoot> root;

/* Previous tonemapper: Unreal Engine FilmToneMap (TonemapCommon.ush) with the default post process
// Film settings. Input is linear Rec.709, output is sRGB encoded.
static const float3x3 sRGB_2_AP1 = {
    0.6130974, 0.3395231, 0.0473795,
    0.0701937, 0.9163539, 0.0134524,
    0.0206156, 0.1095698, 0.8698146,
};
static const float3x3 AP1_2_sRGB = {
    1.7050510, -0.6217920, -0.0832590,
    -0.1302564, 1.1408047, -0.0105483,
    -0.0240034, -0.1289690, 1.1529724,
};
static const float3x3 AP1_2_AP0 = {
    0.6954522414, 0.1406786965, 0.1638690622,
    0.0447945634, 0.8596711185, 0.0955343182,
    -0.0055258826, 0.0040252103, 1.0015006723,
};
static const float3x3 AP0_2_AP1 = {
    1.4514393161, -0.2365107469, -0.2149285693,
    -0.0765537734, 1.1762296998, -0.0996759264,
    0.0083161484, -0.0060324498, 0.9977163014,
};
static const float3 AP1_LUMINANCE = float3(0.2722287168, 0.6740817658, 0.0536895174);

static const float FilmSlope = 0.88;
static const float FilmToe = 0.55;
static const float FilmShoulder = 0.26;
static const float FilmBlackClip = 0.0;
static const float FilmWhiteClip = 0.04;

float acesSaturation(float3 rgb)
{
    const float minRgb = min(rgb.r, min(rgb.g, rgb.b));
    const float maxRgb = max(rgb.r, max(rgb.g, rgb.b));
    return (max(maxRgb, 1e-10) - max(minRgb, 1e-10)) / max(maxRgb, 1e-2);
}

float acesHue(float3 rgb)
{
    if (rgb.r == rgb.g && rgb.g == rgb.b) return 0.0;
    const float hue = degrees(atan2(sqrt(3.0) * (rgb.g - rgb.b), 2.0 * rgb.r - rgb.g - rgb.b));
    return hue < 0.0 ? hue + 360.0 : hue;
}

float acesLuminanceChroma(float3 rgb)
{
    const float chroma = sqrt(rgb.b * (rgb.b - rgb.g) + rgb.g * (rgb.g - rgb.r) + rgb.r * (rgb.r - rgb.b));
    return (rgb.r + rgb.g + rgb.b + 1.75 * chroma) / 3.0;
}

float acesSigmoidShaper(float x)
{
    const float t = max(1.0 - abs(0.5 * x), 0.0);
    return 0.5 * (1.0 + sign(x) * (1.0 - t * t));
}

float acesGlowGain(float yc, float gain, float mid)
{
    if (yc <= 2.0 / 3.0 * mid) return gain;
    if (yc >= 2.0 * mid) return 0.0;
    return gain * (mid / yc - 0.5);
}
*/

float3 linearToSrgb(float3 c)
{
    c = saturate(c);
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

/*
float3 unrealFilmToneMap(float3 linearRec709)
{
    float3 ap0 = mul(AP1_2_AP0, mul(sRGB_2_AP1, linearRec709));

    const float saturation = acesSaturation(ap0);
    const float glowShape = acesSigmoidShaper((saturation - 0.4) / 0.2);
    ap0 *= 1.0 + acesGlowGain(acesLuminanceChroma(ap0), 0.05 * glowShape, 0.08);

    const float redScale = 0.82;
    const float redPivot = 0.03;
    const float redWidth = 135.0;
    float centeredHue = acesHue(ap0);
    if (centeredHue > 180.0) centeredHue -= 360.0;
    const float hueWeight = pow(smoothstep(0.0, 1.0, 1.0 - abs(2.0 * centeredHue / redWidth)), 2.0);
    ap0.r += hueWeight * saturation * (redPivot - ap0.r) * (1.0 - redScale);

    float3 working = max(mul(AP0_2_AP1, ap0), 0.0);
    working = lerp(dot(working, AP1_LUMINANCE), working, 0.96);

    const float toeScale = 1.0 + FilmBlackClip - FilmToe;
    const float shoulderScale = 1.0 + FilmWhiteClip - FilmShoulder;
    const float middleGrey = 0.18;

    // FilmToe <= 0.8, so middle grey lies on the toe segment.
    const float bt = (middleGrey + FilmBlackClip) / toeScale - 1.0;
    const float toeMatch = log10(middleGrey) - 0.5 * log((1.0 + bt) / (1.0 - bt)) * (toeScale / FilmSlope);
    const float straightMatch = (1.0 - FilmToe) / FilmSlope - toeMatch;
    const float shoulderMatch = FilmShoulder / FilmSlope - straightMatch;

    const float3 logColor = log10(working);
    const float3 straight = FilmSlope * (logColor + straightMatch);
    float3 toe = -FilmBlackClip + 2.0 * toeScale / (1.0 + exp(-2.0 * FilmSlope / toeScale * (logColor - toeMatch)));
    float3 shoulder = 1.0 + FilmWhiteClip - 2.0 * shoulderScale / (1.0 + exp(2.0 * FilmSlope / shoulderScale * (logColor - shoulderMatch)));
    toe = select(logColor < toeMatch, toe, straight);
    shoulder = select(logColor > shoulderMatch, shoulder, straight);

    float3 t = saturate((logColor - toeMatch) / (shoulderMatch - toeMatch));
    if (shoulderMatch < toeMatch) t = 1.0 - t;
    t = (3.0 - 2.0 * t) * t * t;
    float3 tone = lerp(toe, shoulder, t);

    tone = lerp(dot(tone, AP1_LUMINANCE), tone, 0.93);
    return linearToSrgb(mul(AP1_2_sRGB, tone));
}
*/

/* Previous tonemapper: AgX Punchy
// AgX (Troy Sobotka) with the "Punchy" look. Polynomial approximation of the
// default contrast curve by Benjamin Wrensch. Input is linear Rec.709, output
// is sRGB encoded.
static const float3x3 AGX_INSET = {
    0.842479062253094, 0.0784335999999992, 0.0792237451477643,
    0.0423282422610123, 0.878468636469772, 0.0791661274605434,
    0.0423756549057051, 0.0784336, 0.879142973793104,
};
static const float3x3 AGX_OUTSET = {
    1.19687900512017, -0.0980208811401368, -0.0990297440797205,
    -0.0528968517574562, 1.15190312990417, -0.0989611768448433,
    -0.0529716355144438, -0.0980434501171241, 1.15107367264116,
};
static const float AGX_MIN_EV = -12.47393;
static const float AGX_MAX_EV = 4.026069;
static const float3 REC709_LUMINANCE = float3(0.2126, 0.7152, 0.0722);

float3 agxContrast(float3 x)
{
    const float3 x2 = x * x;
    const float3 x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

float3 agxPunchyToneMap(float3 linearRec709)
{
    float3 value = mul(AGX_INSET, linearRec709);
    value = clamp(log2(max(value, 1e-10)), AGX_MIN_EV, AGX_MAX_EV);
    value = agxContrast((value - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV));

    const float luma = dot(value, REC709_LUMINANCE);
    value = pow(max(value, 0.0), 1.35);
    value = luma + 1.4 * (value - luma);

    return linearToSrgb(pow(max(mul(AGX_OUTSET, value), 0.0), 2.2));
}
*/

// ACES filmic curve (Krzysztof Narkowicz 2015 fit of the ACES RRT+ODT), applied per
// channel so highlights keep their saturation. Input is linear Rec.709,
// output is sRGB encoded.
float3 acesFilmicToneMap(float3 linearRec709)
{
    const float3 x = linearRec709 * 0.6;
    const float3 mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    return linearToSrgb(mapped);
}

float3 hashIdToColor(uint id)
{
    // Invalid / background pixel
    if (id == ~0u)
        return float3(0.0, 0.0, 0.0);

    // Offset valid IDs so object ID 0 does not hash to black.
    uint h = (id + 1u) * 2654435761u;
    h ^= h >> 16;

    return float3(
        float((h >> 16) & 0xFFu) / 255.0,
        float((h >>  8) & 0xFFu) / 255.0,
        float( h        & 0xFFu) / 255.0);
}

float3 magma(float value)
{
    static const float3 colors[5] = {
        float3(0.001462, 0.000466, 0.013866),
        float3(0.316654, 0.071690, 0.485380),
        float3(0.716387, 0.214982, 0.475290),
        float3(0.986700, 0.535582, 0.382210),
        float3(0.987053, 0.991438, 0.749504),
    };
    const float scaled = saturate(value) * 4.0;
    const uint segment = min(uint(scaled), 3u);
    return lerp(colors[segment], colors[segment + 1u], scaled - float(segment));
}

// Selection outline geometry, in output pixels.
static const float OutlineWidth = 3.0;
// ceil(3.5 * OutlineSigma): the reach of the mask filter below.
static const int OutlineSearch = 5;
// Width of the Gaussian the selection mask is filtered with, in output pixels.
// It has to span the steps a nearest resample of the render-resolution
// cryptomatte leaves in the mask (up to the upscale ratio), and 3.5 sigma of
// it has to fit inside the OutlineSearch window.
static const float OutlineSigma = 1.3;

// Inverse error function (Giles, "Approximating the erfinv function", 2010),
// accurate to single precision over (-1, 1).
float erfinv(float x)
{
    float w = -log((1.0 - x) * (1.0 + x));
    float p;
    if (w < 5.0)
    {
        w -= 2.5;
        p = 2.81022636e-08;
        p = 3.43273939e-07 + p * w;
        p = -3.5233877e-06 + p * w;
        p = -4.39150654e-06 + p * w;
        p = 0.00021858087 + p * w;
        p = -0.00125372503 + p * w;
        p = -0.00417768164 + p * w;
        p = 0.246640727 + p * w;
        p = 1.50140941 + p * w;
    }
    else
    {
        w = sqrt(w) - 3.0;
        p = -0.000200214257;
        p = 0.000100950558 + p * w;
        p = 0.00134934322 + p * w;
        p = -0.00367342844 + p * w;
        p = 0.00573950773 + p * w;
        p = -0.0076224613 + p * w;
        p = 0.00943887047 + p * w;
        p = 1.00167406 + p * w;
        p = 2.83297682 + p * w;
    }
    return p * x;
}

// The reach of the outline's distance field, in output pixels. A pixel with no
// selection within the mask filter's window sits exactly here.
float outlineReach(float upscaleRatio)
{
    return OutlineWidth + clamp(upscaleRatio, 1.0, 3.0) * 0.5;
}

// Where the output pixels' cryptomatte comes from: the render-resolution AOV,
// nearest resampled. Render pixel p sampled the scene at p + 0.5 + jitter in
// render pixels, so output pixel o looks up the render pixel nearest
// (o + 0.5) * renderSize / outputSize - 0.5 - jitter. Keeping the IDs exact
// is what the outline's mask filter relies on.
struct CryptomatteMap
{
    float2 outputSize;
    float2 renderSize;
    float2 jitter;
};

uint cryptomatteAt(RWTexture2D<uint> idImage, CryptomatteMap map, int2 pixel)
{
    const float2 source = (float2(pixel) + 0.5) * map.renderSize / map.outputSize - 0.5
        - map.jitter;
    return idImage[uint2(clamp(round(source), float2(0.0), map.renderSize - 1.0))];
}

// The selection mask around a thread group: its pixels and OutlineSearch more
// on every side, as 1 where the cryptomatte holds the selected ID. Outside the
// render counts as unselected, so a selection running off the edge is outlined
// along it rather than left open.
static const int OutlineTile = GROUP_SIZE + 2 * OutlineSearch;
groupshared uint outlineSelected[OutlineTile * OutlineTile];
groupshared uint outlineAnySelected;

// Fills the tile for the group `groupId`. Every thread of the group must call
// it, outside the render included: it synchronizes the group.
void loadOutlineTile(RWTexture2D<uint> idImage, CryptomatteMap map, uint2 groupId,
    uint threadIndex, uint selectedId, int2 size)
{
    if (threadIndex == 0u)
        outlineAnySelected = 0u;
    GroupMemoryBarrierWithGroupSync();
    const int2 origin = int2(groupId) * GROUP_SIZE - OutlineSearch;
    [loop]
    for (uint i = threadIndex; i < uint(OutlineTile * OutlineTile); i += GROUP_SIZE * GROUP_SIZE)
    {
        const int2 samplePixel = origin + int2(int(i) % OutlineTile, int(i) / OutlineTile);
        const bool selected = all(samplePixel >= int2(0)) && all(samplePixel < size)
            && cryptomatteAt(idImage, map, samplePixel) == selectedId;
        outlineSelected[i] = selected ? 1u : 0u;
        if (selected)
            InterlockedOr(outlineAnySelected, 1u);
    }
    GroupMemoryBarrierWithGroupSync();
}

// This frame's estimate of the signed distance, in output pixels, from the
// pixel at `local` in its group to the edge of the selection. Negative inside
// it.
//
// The cryptomatte is a nearest resample of the render-resolution AOV, so on its
// own the selection mask is a staircase with steps of up to the
// upscale ratio. Distances measured to that staircase inherit it. Instead the
// mask is filtered with a Gaussian: across a straight edge the filtered value
// is 0.5 * erfc(d / (sigma * sqrt(2))) of the distance d to the edge, which is
// inverted exactly. The staircase averages out inside the filter, so the edge
// comes out near its sub-texel position on the first frame. What remains is a
// small zero-mean error that moves with the jitter, which the caller's running
// average removes.
float outlineDistance(int2 local, float upscaleRatio)
{
    const float reach = outlineReach(upscaleRatio);
    // No selection near the group: the filtered mask is zero everywhere in it,
    // which the clamp below turns into the reach.
    if (outlineAnySelected == 0u)
        return reach;
    const float window = 3.5 * OutlineSigma;
    // The Gaussian is separable; only the disc it is cut to is not.
    float gaussian[OutlineSearch + 1];
    [unroll]
    for (int i = 0; i <= OutlineSearch; ++i)
        gaussian[i] = exp(-float(i * i) / (2.0 * OutlineSigma * OutlineSigma));
    float selectedWeight = 0.0;
    float totalWeight = 0.0;
    [loop]
    for (int y = -OutlineSearch; y <= OutlineSearch; ++y)
    {
        [loop]
        for (int x = -OutlineSearch; x <= OutlineSearch; ++x)
        {
            if (float(x * x + y * y) > window * window)
                continue;
            const float weight = gaussian[abs(x)] * gaussian[abs(y)];
            totalWeight += weight;
            if (outlineSelected[(local.y + OutlineSearch + y) * OutlineTile
                    + local.x + OutlineSearch + x] != 0u)
                selectedWeight += weight;
        }
    }
    // The clamp keeps a pixel far from any selection from producing an
    // infinite distance and dragging the average around with it.
    const float fraction = clamp(selectedWeight / totalWeight, 1.0e-6, 1.0 - 1.0e-6);
    const float signedDistance = OutlineSigma * sqrt(2.0) * erfinv(1.0 - 2.0 * fraction);
    return clamp(signedDistance, -reach, reach);
}

// The outline's coverage at a settled distance: a ring hugging the silhouette
// from outside, faded at both of its edges. A one-texel selection - a Gaussian
// footprint - never reaches the inside of the ring and so sits within it.
float outlineCoverage(float signedDistance, float upscaleRatio)
{
    const float aa = clamp(upscaleRatio, 1.0, 3.0) * 0.5;
    return smoothstep(-aa, aa, signedDistance)
        * (1.0 - smoothstep(OutlineWidth - aa, OutlineWidth + aa, signedDistance));
}

[shader("compute")]
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void main(uint3 threadId : SV_DispatchThreadID, uint3 groupId : SV_GroupID,
    uint3 localId : SV_GroupThreadID, uint threadIndex : SV_GroupIndex)
{
    const ViewportCompositePushConstants pushConstants = *root.arguments;
    RWTexture2D<float4> colorImage = ResourceDescriptorHeap[pushConstants.colorImage];
    RWTexture2D<float4> outputImage = ResourceDescriptorHeap[pushConstants.outputImage];
    RWTexture2D<uint> idImage = ResourceDescriptorHeap[pushConstants.idImage];
    // Not every renderer writes the albedo, normal and position images, so
    // only their own views fetch their descriptors.
    // The images may be allocated larger than the render. Only the logical
    // rectangle is traced.
    const uint width = pushConstants.width;
    const uint height = pushConstants.height;
    uint imageWidth;
    uint imageHeight;
    outputImage.GetDimensions(imageWidth, imageHeight);
    // The selection outline of the beauty view reads the mask around the whole
    // group, so every thread joins in filling it before any may leave.
    CryptomatteMap cryptomatte;
    cryptomatte.outputSize = float2(width, height);
    cryptomatte.renderSize = float2(pushConstants.cryptomatteWidth,
        pushConstants.cryptomatteHeight);
    cryptomatte.jitter = float2(pushConstants.cryptomatteJitterX,
        pushConstants.cryptomatteJitterY);
    const bool outlined = pushConstants.bufferVisualization == 0
        && pushConstants.selectedCryptomatteId != ~0u;
    if (outlined)
        loadOutlineTile(idImage, cryptomatte, groupId.xy, threadIndex,
            pushConstants.selectedCryptomatteId, int2(width, height));
    // One extra column and row past the logical edge repeat the edge pixel,
    // so a linear sampler reading right at the edge does not blend in stale
    // texels from an earlier, larger render.
    if (threadId.x > width || threadId.y > height
        || threadId.x >= imageWidth || threadId.y >= imageHeight)
        return;

    const int2 pixel = int2(min(threadId.xy, uint2(width, height) - 1u));
    float3 color;
    float  alpha = 1.0;

    const bool beautyView = pushConstants.bufferVisualization == 0;
    if (beautyView)
    {
        const float4 hdr = colorImage[pixel];
        alpha = clamp(hdr.a, 0.0, 1.0);
        const float3 straight = alpha > 0.0
            ? hdr.rgb / alpha * exp2(clamp(pushConstants.exposure, -20.0, 20.0))
            : float3(0.0);
        color = pushConstants.tonemappingEnabled != 0 ? acesFilmicToneMap(straight) : straight;

        if (outlined)
        {
            RWTexture2D<float> selectionSdf =
                ResourceDescriptorHeap[pushConstants.selectionSdfImage];
            // The extra edge column measures from its own position; there the
            // render counts as unselected, as it does past the edge pixel.
            const float current = outlineDistance(int2(localId.xy), pushConstants.upscaleRatio);
            // Running average of every frame since the outline history
            // restarted, independent of beauty accumulation. Only this
            // thread's own texel is read and written here, while the
            // neighbourhood above comes from an image no thread in this
            // dispatch writes, so the blend needs no second pass.
            const uint count = pushConstants.selectionSampleCount;
            const float averaged = count == 0u ? current
                : lerp(selectionSdf[threadId.xy], current, 1.0 / float(count + 1u));
            selectionSdf[threadId.xy] = averaged;
            // Blended, not overwritten: the fractional coverage at the ring's
            // edges is what anti-aliases it against the beauty behind it.
            const float coverage = outlineCoverage(averaged, pushConstants.upscaleRatio);
            color = lerp(color, ViewportSelectionColor, coverage);
            alpha = max(alpha, coverage);
        }
    }
    else switch (pushConstants.bufferVisualization)
    {
    case 1: // Albedo
    {
        RWTexture2D<float4> albedoImage = ResourceDescriptorHeap[pushConstants.albedoImage];
        const float4 s = albedoImage[pixel];
        color = s.rgb;
        alpha = s.a;
        break;
    }
    case 2: // Normal — remap [-1, 1] → [0, 1]
    {
        RWTexture2D<float4> normalImage = ResourceDescriptorHeap[pushConstants.normalImage];
        const float3 n = normalImage[pixel].xyz;
        color = n * 0.5 + 0.5;
        break;
    }
    case 3: // Cryptomatte — hash ID to color
    {
        color = hashIdToColor(cryptomatteAt(idImage, cryptomatte, pixel));
        break;
    }
    case 4: // Position — visualize as absolute world-space distance
    {
        RWTexture2D<float4> positionImage = ResourceDescriptorHeap[pushConstants.positionImage];
        const float3 p = positionImage[pixel].xyz;
        color = abs(p) / (abs(p) + 1.0); // Reinhard-like to keep it in [0,1]
        break;
    }
    default: // Beauty is handled above; keep invalid values black.
    {
        color = float3(0.0);
        alpha = 1.0;
        break;
    }
    }

    // RmlUi's Vulkan blend state expects premultiplied color. Keeping the
    // viewport texture in that form lets transparent-background renders show
    // the viewport panel behind the image without dark fringes.
    outputImage[threadId.xy] = float4(color * alpha, alpha);
}
