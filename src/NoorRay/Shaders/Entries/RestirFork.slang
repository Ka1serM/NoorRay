import NoorRHI;
import Restir.Bridge;
import Restir.Passes;
import Wavefront.Records;

//! Starts the forks of the pixels of a sweep, each in the pool slot of its index in the sweep.
[shader("compute")]
[numthreads(64, 1, 1)]
void restirFork(uint3 id : SV_DispatchThreadID) {
    const RestirArguments arguments = rootArguments<RestirArguments>();
    const RestirFrame frame = arguments.frame[0];
    beginRestir(frame);
    const uint threads = frame.wavefront[0].queues.gridThreads;
    for (uint index = id.x; index < arguments.count; index += threads)
        startRestirFork(frame, arguments.parity, arguments.start + index, index);
}
