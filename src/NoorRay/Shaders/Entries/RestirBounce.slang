import NoorRHI;
import Restir.Bridge;
import Restir.Passes;
import Wavefront.Records;
import Wavefront.PathState;
import Wavefront.Queues;

//! The initial paths at their hits and misses, which the bins hold in one range from the first bin on; in the finish mode, the paths still in flight.
[shader("compute")]
[numthreads(64, 1, 1)]
void restirBounce(uint3 id : SV_DispatchThreadID) {
    const RestirArguments arguments = rootArguments<RestirArguments>();
    const RestirFrame frame = arguments.frame[0];
    beginRestir(frame);
    const WavefrontFrame wavefront = frame.wavefront[0];
    const QueueRecord queues = wavefront.queues;
    if (arguments.mode == restirPathFinish) {
        if (id.x == 0)
            atomicStore(frame.counters, 0u);
        const uint count = activeCount(wavefront);
        for (uint index = id.x; index < count; index += queues.gridThreads)
            finishInFlightInitialPath(frame, activeQueue(queues, arguments.parity)[index]);
        return;
    }
    const uint count = atomicLoad(queues.binCursors + (queues.binCount - 1));
    for (uint index = id.x; index < count; index += queues.gridThreads)
        restirBouncePath(frame, arguments.parity, queues.shade[index]);
}
