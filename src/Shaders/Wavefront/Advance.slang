#ifndef RAYTRACING_ADVANCE_SLANG
#define RAYTRACING_ADVANCE_SLANG

#include "../Shared.h"
#include "../Bindings.slang"

[[vk::push_constant]] ConstantBuffer<PushData> pushConstants;

#include "../Common.slang"

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    dispatchThreadID = dispatchThreadID;
    queueCounters[QUEUE_COUNTER_ACTIVE] = queueCounters[QUEUE_COUNTER_NEXT];
    queueCounters[QUEUE_COUNTER_NEXT] = 0;
    queueCounters[QUEUE_COUNTER_HIT] = 0;
    queueCounters[QUEUE_COUNTER_SHADOW] = 0;
}

#endif
