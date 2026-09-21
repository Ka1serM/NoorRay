#include "RealtimeArgs.slang"

static const uint AccumulateGroupSize = 8u;

// Folds the final output beauty into the running average, as the offline
// tracer does, so a still view converges. Sample 0 restarts the average.
[shader("compute")]
[numthreads(AccumulateGroupSize, AccumulateGroupSize, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    const Frame frame = realtimeArgs()->frame;
    const RealtimeView view = realtimeArgs()->view;
    if (threadId.x >= view.outputWidth || threadId.y >= view.outputHeight)
        return;
    const uint2 output = threadId.xy;
    RWTexture2D<float4> color = ResourceDescriptorHeap[frame.colorImage];
    float4* accumulation = frame.accumulation;
    const uint pixel = output.y * view.outputWidth + output.x;
    const float4 sum = color[output]
        + (frame.sampleIndex == 0u ? float4(0.0) : accumulation[pixel]);
    accumulation[pixel] = sum;
    color[output] = sum / float(frame.sampleIndex + 1u);
}
