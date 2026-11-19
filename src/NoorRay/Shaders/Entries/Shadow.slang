import NoorRHI;
import Wavefront.Records;
import Wavefront.Queues;
import Wavefront.Shadow;

[shader("compute")]
[numthreads(64, 1, 1)]
void shadow(uint3 id : SV_DispatchThreadID) {
    const WavefrontArguments arguments = rootArguments<WavefrontArguments>();
    const WavefrontFrame frame = arguments.frame[0];
    const uint count = shadowCount(frame);
    for (uint index = id.x; index < count; index += frame.queues.gridThreads)
        traceShadow(frame, arguments.parity, arguments.bin, index);
}
