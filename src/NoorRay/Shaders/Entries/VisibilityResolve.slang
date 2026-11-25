import NoorRHI;
import Wavefront.Records;
import Wavefront.VisibilityCache;

//! One thread per cell of the visibility cache; bin 1 clears the cache.
[shader("compute")]
[numthreads(64, 1, 1)]
void visibilityResolve(uint3 id : SV_DispatchThreadID) {
    const WavefrontArguments arguments = rootArguments<WavefrontArguments>();
    const VisibilityCacheRecord cache = arguments.frame[0].visibility;
    if (id.x > cache.mask)
        return;
    resolveVisibility(cache, arguments.frame[0].scene[0], id.x, arguments.bin != 0);
}
