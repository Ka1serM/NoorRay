import NoorRHI;
import Wavefront.Records;
import Wavefront.Queues;
import Wavefront.Generate;

[shader("compute")]
[numthreads(64, 1, 1)]
void generate(uint3 id : SV_DispatchThreadID) {
    const WavefrontArguments arguments = rootArguments<WavefrontArguments>();
    const WavefrontFrame frame = arguments.frame[0];
    const uint count = activeCount(frame);
    for (uint index = id.x; index < count; index += frame.queues.gridThreads)
        generatePath(frame, index);
}
