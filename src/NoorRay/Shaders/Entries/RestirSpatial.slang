import NoorRHI;
import Restir.Bridge;
import Restir.Passes;

[shader("compute")]
[numthreads(64, 1, 1)]
void restirSpatial(uint3 id : SV_DispatchThreadID) {
    const RestirFrame frame = rootArguments<RestirArguments>().frame[0];
    const uint pixels = frame.width * frame.height;
    for (uint pixel = id.x; pixel < pixels; pixel += frame.gridThreads)
        resampleRestirSpatial(frame, pixel);
}
