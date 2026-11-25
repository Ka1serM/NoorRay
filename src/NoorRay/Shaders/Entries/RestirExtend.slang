import NoorRHI;
import Restir.Bridge;
import Restir.Passes;
import Wavefront.Records;
import Wavefront.Queues;

[shader("compute")]
[numthreads(64, 1, 1)]
void restirExtend(uint3 id : SV_DispatchThreadID) {
    const RestirArguments arguments = rootArguments<RestirArguments>();
    const RestirFrame frame = arguments.frame[0];
    const WavefrontFrame wavefront = frame.wavefront[0];
    const uint count = activeCount(wavefront);
    for (uint index = id.x; index < count; index += wavefront.queues.gridThreads)
        restirExtendPath(frame, arguments.parity, arguments.mode, index);
}
