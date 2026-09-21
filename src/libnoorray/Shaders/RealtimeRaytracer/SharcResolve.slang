#include "Sharc.slang"

static const uint SharcResolveGroupSize = 256u;

// SHaRC resolve: folds this frame's accumulated samples into the resolved
// cache, applies temporal accumulation and evicts stale entries.
[shader("compute")]
[numthreads(SharcResolveGroupSize, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    if (!sharcEnabled() || threadId.x >= realtimeArgs()->radianceCache.capacity)
        return;

    SharcResolveParameters resolve;
    resolve.cameraPositionPrev = realtimeArgs()->view.previousCameraPosition;
    resolve.accumulationFrameNum = realtimeArgs()->radianceCache.accumulationFrames;
    resolve.responsiveFrameNum = realtimeArgs()->radianceCache.accumulationFrames;
    resolve.staleFrameNumMax = realtimeArgs()->radianceCache.staleFrames;
    resolve.frameIndex = realtimeArgs()->view.frameIndex;
    SharcResolveEntry(threadId.x, makeSharcParameters(), resolve);
}
