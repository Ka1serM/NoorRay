import NoorRHI;
import Restir.Bridge;
import Restir.Passes;
import Wavefront.Records;

//! One stage of the temporal or spatial resampling of a sweep of pixels; thread 0 empties the pending list that the next stage writes.
[shader("compute")]
[numthreads(64, 1, 1)]
void restirResample(uint3 id : SV_DispatchThreadID) {
    const RestirArguments arguments = rootArguments<RestirArguments>();
    const RestirFrame frame = arguments.frame[0];
    beginRestir(frame);
    if (id.x == 0)
        atomicStore(frame.counters + (arguments.stage + 1) % restirStages, 0u);
    const uint count = resampleCount(frame, arguments);
    for (uint index = id.x; index < count; index += frame.wavefront[0].queues.gridThreads)
        resamplePixel(frame, arguments, index);
}
