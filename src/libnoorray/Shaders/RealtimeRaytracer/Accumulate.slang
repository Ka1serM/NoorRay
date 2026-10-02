#include "RealtimeArgs.slang"

static const uint AccumulateGroupSize = 8u;

// Folds the final output beauty into the running average, as the offline
// tracer does, so a still view converges. Sample 0 or an animated hit this
// frame restarts the average.
[shader("compute")]
[numthreads(AccumulateGroupSize, AccumulateGroupSize, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    const Frame frame = realtimeArgs()->frame;
    const RealtimeView view = realtimeArgs()->view;
    if (threadId.x >= view.outputWidth || threadId.y >= view.outputHeight)
        return;
    const uint2 output = threadId.xy;
    BeautyAccumulation* state = frame.beautyAccumulation;
    const uint slot = frame.accumulationSlot;
    const bool restart = frame.sampleIndex == 0u || state->animatedHit[slot] != 0u;
    const uint previousSamples = restart ? 0u : state->sampleCount[slot];
    if (all(output == 0u))
    {
        state->animatedHit[slot ^ 1u] = 0u;
        state->sampleCount[slot ^ 1u] = previousSamples + 1u;
    }
    RWTexture2D<float4> color = ResourceDescriptorHeap[frame.colorImage];
    float4* accumulation = frame.accumulation;
    const uint pixel = output.y * view.outputWidth + output.x;
    const float4 sum = color[output] + (restart ? float4(0.0) : accumulation[pixel]);
    accumulation[pixel] = sum;
    color[output] = sum / float(previousSamples + 1u);
}
