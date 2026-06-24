#ifndef RAYTRACING_CONNECT_SLANG
#define RAYTRACING_CONNECT_SLANG

#include "../Shared.h"
#include "../Bindings.slang"

[[vk::push_constant]] ConstantBuffer<PushData> pushConstants;

#include "../Common.slang"

// Shadow-ray visibility test and radiance accumulation.
//
// Each path emits exactly ONE shadow ray per bounce (queueNEEShadowRay), so
// every ShadowWorkItem has a unique SampleIndex at this stage.  There is no
// concurrent write to the same pathStates slot — the race condition that
// existed when multiple shadow rays shared a path index is structurally
// impossible here.
//
// Contribution is already pre-multiplied by the path throughput; no read of
// pathStates.Throughput is required.

[shader("compute")]
[numthreads(QUEUE_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    while (true) {
        uint workIndex;
        if (!popQueueIndex(QUEUE_COUNTER_SHADOW, workIndex))
            return;

        ShadowWorkItem shadow = shadowQueue[workIndex];
        if (!traceShadowPath(shadow.Origin, shadow.Direction, shadow.TMin, shadow.TMax)) {
            PathState state = pathStates[shadow.SampleIndex];
            state.Radiance += shadow.Contribution;
            pathStates[shadow.SampleIndex] = state;
        }
    }
}

#endif // RAYTRACING_CONNECT_SLANG
