#pragma once

#include "../../Shared/Viewport.h"

static const int GROUP_SIZE = 16;

[[vk::push_constant]] ConstantBuffer<ViewportCompositeRoot> root;

// Blender 5.2 "Raw" with the "Very Low Contrast" Filmic look.  The OCIO
// config defines that look in Filmic Log as filmic_to_0-35_1-30.spi1d
// followed by the inverse of filmic_to_0-70_1-03.spi1d.  These samples are
// that exact LUT composition, resampled uniformly in Filmic-log space.
static const float BlenderVeryLowContrast[65] = {
    0.000000000, 0.056577142, 0.097656381, 0.130399487, 0.157937676, 0.181923780, 0.203337549, 0.222808383,
    0.240764739, 0.257511610, 0.273274335, 0.288223848, 0.302493459, 0.316189037, 0.329396276, 0.342185631,
    0.354615956, 0.366736906, 0.378591012, 0.390215104, 0.401641265, 0.412898023, 0.424010717, 0.435002284,
    0.445893610, 0.456703839, 0.467450892, 0.478151559, 0.488821724, 0.499476748, 0.510131490, 0.520800442,
    0.531498120, 0.542238995, 0.553037757, 0.563909461, 0.574869708, 0.585934819, 0.597122104, 0.608205789,
    0.618567634, 0.629090756, 0.639786767, 0.650668375, 0.661749521, 0.673045596, 0.684573639, 0.696352601,
    0.708403699, 0.720750912, 0.733421386, 0.746446325, 0.759861901, 0.773710598, 0.788043112, 0.802921234,
    0.818421886, 0.834643622, 0.851717823, 0.869827411, 0.889243604, 0.910402333, 0.934093971, 0.962054729,
    1.000000000
};

float blenderVeryLowContrast1(float value)
{
    if (value <= 0.0) return 0.0;
    // Filmic Log's effective scene-linear range from Blender's config: the
    // uniform [0, 0.66] allocation maps the first 16.5 stops of the 25-stop
    // lg2 allocation [-12.473931188, 12.526068812] onto [0, 1].
    const float x = clamp((log2(value) + 12.473931188) / 16.5, 0.0, 1.0);
    const float p = x * 64.0;
    const uint i = min((uint)floor(p), 63u);
    const float y = lerp(BlenderVeryLowContrast[i], BlenderVeryLowContrast[i + 1u], p - float(i));
    return exp2(y * 16.5 - 12.473931188);
}

float3 blenderRawVeryLowContrast(float3 color)
{
    return clamp(float3(
        blenderVeryLowContrast1(color.r),
        blenderVeryLowContrast1(color.g),
        blenderVeryLowContrast1(color.b)), 0.0, 1.0);
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

// This frame's estimate of the signed distance, in output pixels, from `pixel`
// to the edge of the selection. Negative inside it.
//
// The cryptomatte this reads is a nearest resample of the render-resolution
// AOV, so on its own the selection mask is a staircase with steps of up to the
// upscale ratio. Distances measured to that staircase inherit it. Instead the
// mask is filtered with a Gaussian: across a straight edge the filtered value
// is 0.5 * erfc(d / (sigma * sqrt(2))) of the distance d to the edge, which is
// inverted exactly. The staircase averages out inside the filter, so the edge
// comes out near its sub-texel position on the first frame. What remains is a
// small zero-mean error that moves with the jitter, which the caller's running
// average removes.
float outlineDistance(RWTexture2D<uint> idImage, int2 pixel, uint selectedId,
    int2 size, float upscaleRatio)
{
    const float aa = clamp(upscaleRatio, 1.0, 3.0) * 0.5;
    const float reach = OutlineWidth + aa;
    const float window = 3.5 * OutlineSigma;
    float selectedWeight = 0.0;
    float totalWeight = 0.0;
    // Kept as real loops. The bounds are compile-time constants with no early
    // exit, so both would otherwise unroll into 121 inlined bindless image
    // loads, and a body that size is what NVIDIA's SPIR-V compiler was
    // crashing on. Rolled up it is also less instruction cache for no
    // measurable difference: the cost here is the texture traffic.
    [loop]
    for (int y = -OutlineSearch; y <= OutlineSearch; ++y)
    {
        [loop]
        for (int x = -OutlineSearch; x <= OutlineSearch; ++x)
        {
            const float distanceSquared = float(x * x + y * y);
            if (distanceSquared > window * window)
                continue;
            const int2 samplePixel = pixel + int2(x, y);
            // Outside the render counts as unselected, so a selection running
            // off the edge is outlined along it rather than left open.
            const bool selected = all(samplePixel >= int2(0))
                && all(samplePixel < size)
                && idImage[samplePixel] == selectedId;
            const float weight = exp(-distanceSquared
                / (2.0 * OutlineSigma * OutlineSigma));
            totalWeight += weight;
            selectedWeight += selected ? weight : 0.0;
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
void main(uint3 threadId : SV_DispatchThreadID)
{
    const ViewportCompositePushConstants pushConstants = *root.arguments;
    RWTexture2D<float4> colorImage = ResourceDescriptorHeap[pushConstants.colorImage];
    RWTexture2D<float4> outputImage = ResourceDescriptorHeap[pushConstants.outputImage];
    RWTexture2D<uint> idImage = ResourceDescriptorHeap[pushConstants.idImage];
    // Not every renderer writes the albedo, normal and position images, so
    // only their own views fetch their descriptors.
    uint* overdrawImage = pushConstants.overdraw;
    // The images may be allocated larger than the render. Only the logical
    // rectangle is traced, and the overdraw buffer uses its width as stride.
    const uint width = pushConstants.width;
    const uint height = pushConstants.height;
    uint imageWidth;
    uint imageHeight;
    outputImage.GetDimensions(imageWidth, imageHeight);
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
        color = pushConstants.tonemappingEnabled != 0 ? blenderRawVeryLowContrast(straight) : straight;

        if (pushConstants.selectedCryptomatteId != ~0u)
        {
            RWTexture2D<float> selectionSdf =
                ResourceDescriptorHeap[pushConstants.selectionSdfImage];
            const float current = outlineDistance(idImage, pixel,
                pushConstants.selectedCryptomatteId, int2(width, height),
                pushConstants.upscaleRatio);
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
        color = hashIdToColor(idImage[pixel]);
        break;
    }
    case 4: // Position — visualize as absolute world-space distance
    {
        RWTexture2D<float4> positionImage = ResourceDescriptorHeap[pushConstants.positionImage];
        const float3 p = positionImage[pixel].xyz;
        color = abs(p) / (abs(p) + 1.0); // Reinhard-like to keep it in [0,1]
        break;
    }
    case 5: // Gaussian proxy overdraw view.
    {
        const float count = float(overdrawImage[uint(pixel.y) * width + uint(pixel.x)]);
        const float range = float(max(pushConstants.overdrawMax, 1u));
        color = magma(log2(count + 1.0) / log2(range + 1.0));
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
