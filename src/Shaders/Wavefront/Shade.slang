#ifndef RAYTRACING_SHADE_SLANG
#define RAYTRACING_SHADE_SLANG

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
        if (!popQueueIndex(QUEUE_COUNTER_HIT, workIndex))
            return;

        HitWorkItem hitWorkItem = hitQueue[workIndex];
        uint sampleIndex = hitWorkItem.SampleIndex;
        PathState state = pathStates[sampleIndex];
        float2 hitBary = hitWorkItem.Barycentrics;
        SurfaceData surface = loadComputeSurfaceData(
            hitWorkItem.InstanceIndex,
            hitWorkItem.PrimitiveIndex,
            float3(1.0 - hitBary.x - hitBary.y, hitBary.x, hitBary.y));

        Payload payload;
        payload.initialize(state.RngState);

        if (state.Depth == 0u)
            state.PrimaryObjectIndex = hitWorkItem.InstanceIndex;

        integratorShadeSurface(surface, hitWorkItem.RayDirection, sampleIndex, state, payload);
        pathStates[sampleIndex] = state;
    }
}

#endif
