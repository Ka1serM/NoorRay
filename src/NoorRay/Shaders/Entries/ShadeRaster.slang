import NoorRHI;
import Wavefront.Records;
import Wavefront.Queues;
import Raster.Shade;

[shader("compute")]
[numthreads(64, 1, 1)]
void shadeRaster(uint3 id : SV_DispatchThreadID) {
    const WavefrontArguments arguments = rootArguments<WavefrontArguments>();
    const WavefrontFrame frame = arguments.frame[0];
    const uint count = binnedCount(frame, arguments.bin);
    for (uint index = id.x; index < count; index += frame.queues.gridThreads)
        shadeRasterPath(frame, arguments.parity, arguments.bin, binnedSlot(frame, arguments.bin, index));
}
