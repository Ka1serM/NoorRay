import NoorRHI;
import Wavefront.Records;
import Wavefront.Queues;

[shader("compute")]
[numthreads(1, 1, 1)]
void advance(uint3 id : SV_DispatchThreadID) {
    const WavefrontArguments arguments = rootArguments<WavefrontArguments>();
    advanceQueues(arguments.frame[0], arguments.parity, arguments.bin);
}
