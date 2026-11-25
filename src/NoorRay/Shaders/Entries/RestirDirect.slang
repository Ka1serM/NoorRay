import NoorRHI;
import Restir.Bridge;
import Restir.DirectLighting;
import Wavefront.Records;

[shader("compute")]
[numthreads(64, 1, 1)]
void restirDirect(uint3 id : SV_DispatchThreadID) {
    const RestirArguments arguments = rootArguments<RestirArguments>();
    const RestirFrame frame = arguments.frame[0];
    beginRestir(frame);
    const uint pixels = frame.width * frame.height;
    for (uint index = id.x; index < pixels; index += frame.wavefront[0].queues.gridThreads) {
        if (arguments.mode == restirDirectSample)
            sampleDirect(frame, restirPixel(frame, index));
        else
            resampleDirect(frame, restirPixel(frame, index));
    }
}
