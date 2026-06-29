#pragma once

static const int GROUP_SIZE = 16;

[[vk::binding(0, 0)]] [format("rgba32f")] RWTexture2D<float4> inputImage;
[[vk::binding(1, 0)]] [format("rgba8")] RWTexture2D<float4> outputImage;
[[vk::binding(2, 0)]] [format("r32ui")] RWTexture2D<uint> idImage;

struct PushConstants
{
    uint selectedIndex;
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

[shader("compute")]
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint width;
    uint height;
    outputImage.GetDimensions(width, height);
    if (threadId.x >= width || threadId.y >= height)
        return;
    const float4 hdr = inputImage[int2(threadId.xy)];
    const float alpha = clamp(hdr.a, 0.0, 1.0);
    const float3 straightColor = alpha > 0.0 ? hdr.rgb / alpha : float3(0.0);
    float3 color = acesFitted(straightColor);

    static const int outlineRadius = 3;
    const int2 pixel = int2(threadId.xy);
    if (pushConstants.selectedIndex != ~0u && idImage[pixel] != pushConstants.selectedIndex)
    {
        bool outline = false;
        for (int y = -outlineRadius; y <= outlineRadius && !outline; ++y)
        {
            for (int x = -outlineRadius; x <= outlineRadius; ++x)
            {
                const int2 offset = int2(x, y);
                const int2 samplePixel = pixel + offset;
                if (dot(offset, offset) <= outlineRadius * outlineRadius &&
                    all(samplePixel >= int2(0)) &&
                    all(samplePixel < int2(width, height)) &&
                    idImage[samplePixel] == pushConstants.selectedIndex)
                {
                    outline = true;
                    break;
                }
            }
        }
        if (outline)
            color = float3(1.0, 0.42, 0.05);
    }

    outputImage[pixel] = float4(color, alpha);
}
