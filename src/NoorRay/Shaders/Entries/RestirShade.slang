import NoorRHI;
import Restir.Bridge;
import Restir.Passes;
import Wavefront.Records;

[shader("compute")]
[numthreads(64, 1, 1)]
void restirShade(uint3 id : SV_DispatchThreadID) {
    const RestirFrame frame = rootArguments<RestirArguments>().frame[0];
    beginRestir(frame);
    const uint pixels = frame.width * frame.height;
    for (uint pixel = id.x; pixel < pixels; pixel += frame.wavefront[0].queues.gridThreads)
        shadeRestir(frame, pixel);
}
