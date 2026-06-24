#ifndef RAYTRACING_EXTEND_SLANG
#define RAYTRACING_EXTEND_SLANG

#include "../Shared.h"
#include "../Bindings.slang"

[[vk::push_constant]] ConstantBuffer<PushData> pushConstants;

#include "../Common.slang"
#include "../Integrators.slang"

[shader("compute")]
[numthreads(QUEUE_GROUP_SIZE / 2, 2, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupID : SV_GroupID,
    uint3 groupThreadID : SV_GroupThreadID)
{
    dispatchThreadID = dispatchThreadID;
    groupID = groupID;
    groupThreadID = groupThreadID;
    while (true) {
        uint workIndex;
        if (!popQueueIndex(QUEUE_COUNTER_ACTIVE, workIndex))
            return;

        PathRayWorkItem rayWorkItem = inputRayQueue[workIndex];
        PathState state = pathStates[rayWorkItem.SampleIndex];

        TraceHit hit;
        if (!traceSceneRay(rayWorkItem.Origin, rayWorkItem.Direction, 0.001, 1000.0, hit)) {
            integratorShadeMiss(rayWorkItem.Direction, state);
            pathStates[rayWorkItem.SampleIndex] = state;
            continue;
        }

        HitWorkItem hitWorkItem;
        hitWorkItem.RayDirection = rayWorkItem.Direction;
        hitWorkItem.SampleIndex = rayWorkItem.SampleIndex;
        hitWorkItem.Barycentrics = hit.Barycentrics.yz;
        hitWorkItem.InstanceIndex = hit.InstanceIndex;
        hitWorkItem.PrimitiveIndex = hit.PrimitiveIndex;
        hitWorkItem.Pad0 = 0u;
        appendHit(hitWorkItem);
    }
}

#endif
