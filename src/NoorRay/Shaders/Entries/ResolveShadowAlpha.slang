import NoorRHI;
import Wavefront.Records;
import Wavefront.Queues;
import Wavefront.Shade;

[shader("compute")]
[numthreads(64, 1, 1)]
void resolveShadowAlpha(uint3 id : SV_DispatchThreadID) {
    const WavefrontArguments arguments = rootArguments<WavefrontArguments>();
    const WavefrontFrame frame = arguments.frame[0];
    const uint start = surfaceStart(frame);
    const uint count = surfaceCount(frame);
    for (uint index = id.x; index < count; index += frame.queues.gridThreads)
        resolveShadowAlphaPath(frame, arguments.parity, surfaceSlot(frame, start, index));
}
