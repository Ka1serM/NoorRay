#pragma once

static const int GROUP_SIZE = 16;

[[vk::binding(0, 0)]] [format("rgba32f")] RWTexture2D<float4> colorImage;
[[vk::binding(1, 0)]] [format("rgba8")]   RWTexture2D<float4> outputImage;
[[vk::binding(2, 0)]] [format("r32ui")]   RWTexture2D<uint>   idImage;
[[vk::binding(3, 0)]] [format("rgba8")]   RWTexture2D<float4> albedoImage;
[[vk::binding(4, 0)]] [format("rgba16f")] RWTexture2D<float4> normalImage;
[[vk::binding(5, 0)]] [format("rgba16f")] RWTexture2D<float4> positionImage;
[[vk::binding(6, 0)]] StructuredBuffer<float4> denoisedBuffer;

struct PushConstants
{
    uint  selectedCryptomatteId;
    float exposure;
    int   bufferVisualization;
    int   tonemappingEnabled;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pushConstants;

static const float3x3 ACESInputMat = float3x3(
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777);

static const float3x3 ACESOutputMat = float3x3(
     1.60475, -0.53108, -0.07367,
    -0.10208,  1.10813, -0.00605,
    -0.00327, -0.07276,  1.07602);

float3 rrtAndOdtFit(float3 value)
{
    float3 a = value * (value + 0.0245786) - 0.000090537;
    float3 b = value * (0.983729 * value + 0.4329510) + 0.238081;
    return a / b;
}

float3 acesFitted(float3 color)
{
    color = mul(ACESInputMat, color);
    color = rrtAndOdtFit(color);
    color = mul(ACESOutputMat, color);
    return clamp(color, 0.0, 1.0);
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

[shader("compute")]
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint width;
    uint height;
    outputImage.GetDimensions(width, height);
    if (threadId.x >= width || threadId.y >= height)
        return;

    const int2 pixel = int2(threadId.xy);
    float3 color;
    float  alpha = 1.0;

    const bool beautyView = pushConstants.bufferVisualization == 0
        || pushConstants.bufferVisualization == 5;
    if (beautyView)
    {
        const uint linearPixel = threadId.y * width + threadId.x;
        const float4 hdr = pushConstants.bufferVisualization == 5
            ? denoisedBuffer[linearPixel] : colorImage[pixel];
        alpha = clamp(hdr.a, 0.0, 1.0);
        const float3 straight = alpha > 0.0
            ? hdr.rgb / alpha * exp2(clamp(pushConstants.exposure, -20.0, 20.0))
            : float3(0.0);
        color = pushConstants.tonemappingEnabled != 0 ? acesFitted(straight) : straight;

        static const int outlineRadius = 3;
        const bool selectionEnabled = pushConstants.selectedCryptomatteId != ~0u;
        const bool selectedPixel = selectionEnabled
            && idImage[pixel] == pushConstants.selectedCryptomatteId;
        bool outline = false;
        if (selectedPixel)
        {
            // Gaussian footprints can be only one or a few pixels wide. Mark
            // their boundary directly instead of relying solely on dilation
            // from a neighboring pixel.
            for (int y = -1; y <= 1 && !outline; ++y)
            {
                for (int x = -1; x <= 1; ++x)
                {
                    const int2 offset = int2(x, y);
                    const int2 samplePixel = pixel + offset;
                    if (x == 0 && y == 0)
                        continue;
                    if (!all(samplePixel >= int2(0)) ||
                        !all(samplePixel < int2(width, height)) ||
                        idImage[samplePixel] != pushConstants.selectedCryptomatteId)
                    {
                        outline = true;
                        break;
                    }
                }
            }
        }
        else if (selectionEnabled)
        {
            // Keep the existing soft outer outline for larger selected areas.
            for (int y = -outlineRadius; y <= outlineRadius && !outline; ++y)
            {
                for (int x = -outlineRadius; x <= outlineRadius; ++x)
                {
                    const int2 offset = int2(x, y);
                    const int2 samplePixel = pixel + offset;
                    if (dot(offset, offset) <= outlineRadius * outlineRadius &&
                        all(samplePixel >= int2(0)) &&
                        all(samplePixel < int2(width, height)) &&
                        idImage[samplePixel] == pushConstants.selectedCryptomatteId)
                    {
                        outline = true;
                        break;
                    }
                }
            }
        }
        if (outline)
        {
            color = float3(1.0, 1.00, 0.00);
            alpha = 1.0;
        }
    }
    else switch (pushConstants.bufferVisualization)
    {
    case 1: // Albedo
    {
        const float4 s = albedoImage[pixel];
        color = s.rgb;
        alpha = s.a;
        break;
    }
    case 2: // Normal — remap [-1, 1] → [0, 1]
    {
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

    outputImage[pixel] = float4(color, alpha);
}
