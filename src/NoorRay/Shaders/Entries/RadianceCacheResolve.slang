import NoorRHI;
import Wavefront.RadianceCache;
import Wavefront.Records;

//! bin is the mode: resolve or clear.
[shader("compute")]
[numthreads(64, 1, 1)]
void radianceCacheResolve(uint3 id : SV_DispatchThreadID) {
    const WavefrontArguments arguments = rootArguments<WavefrontArguments>();
    const WavefrontFrame frame = arguments.frame[0];
    const RadianceCacheRecord cache = frame.cache;
    for (uint entry = id.x; entry < cache.capacity; entry += frame.queues.gridThreads)
        resolveRadianceCache(cache, arguments.bin, cache.frameIndex, entry);
}
