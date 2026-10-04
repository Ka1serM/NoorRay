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

// Finishes the output image: resolves its coverage and folds the beauty into
// the running average as the offline tracer does, so a still view converges
// (sample 0 or an animated hit this frame restarts the average).
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

    // Each frame's accumulation state is in one of two slots: its hit shaders
    // mark the slot and this pass resets the other for the next frame, so no
    // thread reads a slot that another resets.
    BeautyAccumulation* state = args.frame.beautyAccumulation;
    const uint slot = args.frame.accumulationSlot;
    const bool restart = args.frame.sampleIndex == 0u || state->animatedHit[slot] != 0u;
    const uint previousSamples = restart ? 0u : state->sampleCount[slot];
    if (all(output == 0u))
    {
        state->animatedHit[slot ^ 1u] = 0u;
        state->sampleCount[slot ^ 1u] = previousSamples + 1u;
    }
    float4* accumulation = args.frame.accumulation;
    const uint pixel = output.y * args.view.outputWidth + output.x;
    if (!restart)
        sum += accumulation[pixel];
    accumulation[pixel] = sum;
    outputColor[output] = sum / float(previousSamples + 1u);
}

#endif
