import NoorRHI;
import Restir.Bridge;
import Restir.Passes;
import Wavefront.Records;
import Wavefront.Queues;

[shader("compute")]
[numthreads(64, 1, 1)]
void restirReplay(uint3 id : SV_DispatchThreadID) {
    const RestirArguments arguments = rootArguments<RestirArguments>();
    const RestirFrame frame = arguments.frame[0];
    beginRestir(frame);
    const QueueRecord queues = frame.wavefront[0].queues;
    const uint count = atomicLoad(queues.binCursors + (queues.binCount - 1));
    for (uint index = id.x; index < count; index += queues.gridThreads)
        restirReplayPath(frame, arguments.parity, queues.shade[index]);
}
