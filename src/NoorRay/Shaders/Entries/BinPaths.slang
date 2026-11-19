import NoorRHI;
import Wavefront.Records;
import Wavefront.Queues;

[shader("compute")]
[numthreads(64, 1, 1)]
void binPaths(uint3 id : SV_DispatchThreadID) {
    const WavefrontArguments arguments = rootArguments<WavefrontArguments>();
    const WavefrontFrame frame = arguments.frame[0];
    const uint count = binSourceCount(frame, arguments.bin);
    for (uint index = id.x; index < count; index += frame.queues.gridThreads)
        sortIntoBin(frame, arguments.parity, arguments.bin, index);
}
