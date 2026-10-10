import noorrhi;

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

// Whether an animated instance can reach the output pixel whose unjittered
// render-space position is `position`. The upscaler's jitter moves its
// bilinear taps by up to half a render pixel either way, so the 3x3 render
// pixels around it cover every frame's taps and the answer does not flicker
// with the jitter.
bool showsAnimatedInstance(float2 position, float2 renderSize)
{
    RealtimeArgs* args = realtimeArgs();
    RWTexture2D<uint> cryptomatte = ResourceDescriptorHeap[args.targets.cryptomatte];
    const Scene scene = args.frame.scene;
    const int2 center = int2(floor(position + 0.5));
    const int2 last = int2(renderSize) - 1;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
        {
            const uint instance = cryptomatte[clamp(center + int2(x, y), int2(0), last)];
            if (instance < scene.instanceCount
                && (scene.instances[instance].flags & InstanceFlagAnimated) != 0u)
                return true;
        }
    return false;
}

// Finishes the output image: resolves its coverage and folds the beauty into
// the running average as the offline tracer does, so a still view converges
// (sample 0 restarts the average; animated instances stay out of it).
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
    // FSR reconstructs RGB but does not preserve the render target's coverage
    // channel. Resolve alpha separately at output resolution so the viewport
    // can composite transparent backgrounds over its own UI background.
    RWTexture2D<float4> outputColor = ResourceDescriptorHeap[args.frame.colorImage];
    float4 sum = outputColor[output];
    sum.a = 1.0;
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
        sum.a = lerp(lerp(a00, a10, f.x), lerp(a01, a11, f.x), f.y);
    }

    // Pixels showing animated instances keep the upscaler's own temporal
    // result; the others average every sample since the camera or scene last
    // changed. Only those changes move the animated pixels, and they restart
    // the whole average, so a pixel that has averaged every sample since then
    // is divided by the frame's sample count.
    float4* accumulation = args.frame.accumulation;
    const uint pixel = output.y * args.view.outputWidth + output.x;
    const uint sampleIndex = args.frame.sampleIndex;
    // An animated pixel's entry holds a negative marker, never a sum of
    // non-negative coverage, until the average restarts.
    const float4 average = sampleIndex == 0u ? float4(0.0) : accumulation[pixel];
    bool animated = average.a < 0.0;
    if (!animated && showsAnimatedInstance(sourcePosition + args.view.jitter, renderSize))
    {
        accumulation[pixel] = float4(-1.0);
        animated = true;
    }
    if (animated)
    {
        outputColor[output] = sum;
        return;
    }
    sum += average;
    accumulation[pixel] = sum;
    outputColor[output] = sum / float(sampleIndex + 1u);
}

#endif
