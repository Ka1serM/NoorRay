import NoorRHI;
import Wavefront.Records;
import Wavefront.Queues;
import Wavefront.Extend;

[shader("compute")]
[numthreads(64, 1, 1)]
void extend(uint3 id : SV_DispatchThreadID) {
    const WavefrontArguments arguments = rootArguments<WavefrontArguments>();
    const WavefrontFrame frame = arguments.frame[0];
    const uint count = activeCount(frame);
    for (uint index = id.x; index < count; index += frame.queues.gridThreads)
        extendPath(frame, arguments.parity, index);
}
