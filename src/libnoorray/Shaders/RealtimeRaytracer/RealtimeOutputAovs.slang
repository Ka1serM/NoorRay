#include "RealtimeArgs.slang"

static const uint RealtimeOutputAovsGroupSize = 8u;

// Resamples the render-resolution first-hit AOVs to the output resolution the
// beauty is upscaled to, so picking, selection outlines and buffer views
// address the same pixels as the beauty image. Nearest sampling keeps
// cryptomatte IDs and positions exact.
[shader("compute")]
[numthreads(RealtimeOutputAovsGroupSize, RealtimeOutputAovsGroupSize, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    RealtimeArgs* args = realtimeArgs();
    const float2 outputSize = float2(args.view.outputWidth, args.view.outputHeight);
    if (threadId.x >= args.view.outputWidth || threadId.y >= args.view.outputHeight)
        return;
    const uint2 output = threadId.xy;
    // Render texel p samples the scene at p + 0.5 + jitter, so the texel
    // holding this output pixel's film position is
    //   p = (output + 0.5) * render / outputSize - 0.5 - jitter.
    // Cancelling the jitter here removes the drift that otherwise sweeps the
    // whole image along the Halton sequence. What it cannot remove is the
    // rounding below: the edge still lands on whichever texel is nearest, and
    // which one that is changes with the jitter. That residual is zero-mean,
    // which is exactly what lets the viewport composite average it out.
    const float2 renderSize = float2(args.frame.width, args.frame.height);
    const float2 sourcePosition = (float2(output) + 0.5) * renderSize / outputSize
        - 0.5 - args.view.jitter;
    const uint2 source = uint2(clamp(round(sourcePosition), float2(0.0), renderSize - 1.0));

    RWTexture2D<float4> albedo = ResourceDescriptorHeap[args.frame.albedoImage];
    RWTexture2D<float4> normal = ResourceDescriptorHeap[args.frame.normalImage];
    RWTexture2D<float4> position = ResourceDescriptorHeap[args.frame.positionImage];
    RWTexture2D<uint> cryptomatte = ResourceDescriptorHeap[args.frame.cryptomatteImage];
    RWTexture2D<float4> renderAlbedo = ResourceDescriptorHeap[args.targets.albedo];
    RWTexture2D<float4> renderNormal = ResourceDescriptorHeap[args.targets.normal];
    RWTexture2D<float4> renderPosition = ResourceDescriptorHeap[args.targets.position];
    RWTexture2D<uint> renderCryptomatte = ResourceDescriptorHeap[args.targets.cryptomatte];

    albedo[output] = renderAlbedo[source];
    normal[output] = renderNormal[source];
    position[output] = renderPosition[source];
    cryptomatte[output] = renderCryptomatte[source];
}
