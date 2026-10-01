#include "RealtimeCamera.slang"

static const uint RealtimeOutputAovsGroupSize = 8u;

// The surface position at a render pixel, from its view Z and camera ray;
// zero where the camera ray found the background.
float4 realtimeSurfacePosition(uint2 pixel)
{
    RealtimeArgs* args = realtimeArgs();
    RWTexture2D<float4> viewZs = ResourceDescriptorHeap[args.targets.viewZ];
    const float viewZ = viewZs[pixel].x;
    if (!(viewZ < args.denoiser.denoisingRange))
        return float4(0.0);
    return float4(realtimeRayPoint(realtimeCameraRay(pixel), viewZ), 1.0);
}

#if defined(REALTIME_PICK)

// One render pixel's surface position, for picking.
[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    RealtimePickRoot* pick = (RealtimePickRoot*)root.arguments;
    pick.position[0] = realtimeSurfacePosition(uint2(pick.pixelX, pick.pixelY));
}

#else

// Resolves the output image's coverage, and resamples the render-resolution
// cryptomatte to the output resolution the beauty is upscaled to, so the
// selection outline and the cryptomatte view address the same pixels as the
// beauty. Nearest sampling keeps the IDs exact.
[shader("compute")]
[numthreads(RealtimeOutputAovsGroupSize, RealtimeOutputAovsGroupSize, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    RealtimeArgs* args = realtimeArgs();
    if (threadId.x >= args.view.outputWidth || threadId.y >= args.view.outputHeight)
        return;
    const uint2 output = threadId.xy;
    const float2 outputSize = float2(args.view.outputWidth, args.view.outputHeight);
    const float2 renderSize = float2(args.frame.width, args.frame.height);
    const float2 sourcePosition = (float2(output) + 0.5) * renderSize / outputSize
        - 0.5 - args.view.jitter;
    const uint2 source = uint2(clamp(round(sourcePosition), float2(0.0), renderSize - 1.0));
    // FSR reconstructs RGB but does not preserve the render target's coverage
    // channel. Resolve alpha separately at output resolution so the viewport
    // can composite transparent backgrounds over its own UI background.
    RWTexture2D<float4> outputColor = ResourceDescriptorHeap[args.frame.colorImage];
    float alpha = 1.0;
    if (args.frame.transparentBackground != 0u)
    {
        RWTexture2D<float4> renderColor = ResourceDescriptorHeap[args.targets.color];
        const float2 p = clamp(sourcePosition, float2(0.0), renderSize - 1.0);
        const int2 p0 = int2(floor(p));
        const int2 p1 = min(p0 + int2(1, 1), int2(args.frame.width, args.frame.height) - 1);
        const float2 f = frac(p);
        const float a00 = renderColor[p0].a;
        const float a10 = renderColor[int2(p1.x, p0.y)].a;
        const float a01 = renderColor[int2(p0.x, p1.y)].a;
        const float a11 = renderColor[p1].a;
        alpha = lerp(lerp(a00, a10, f.x), lerp(a01, a11, f.x), f.y);
    }
    float4 outputPixel = outputColor[output];
    outputPixel.a = alpha;
    outputColor[output] = outputPixel;
    if (args.view.outputCryptomatte != 0u)
    {
        RWTexture2D<uint> dst = ResourceDescriptorHeap[args.frame.cryptomatteImage];
        RWTexture2D<uint> src = ResourceDescriptorHeap[args.targets.cryptomatte];
        dst[output] = src[source];
    }
}

#endif
