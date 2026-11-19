import NoorRHI;
import Wavefront.Records;
import Wavefront.Generate;

[shader("compute")]
[numthreads(64, 1, 1)]
void buildSamples(uint3 id : SV_DispatchThreadID) {
    const WavefrontArguments arguments = rootArguments<WavefrontArguments>();
    const WavefrontFrame frame = arguments.frame[0];
    //! The fixed grid strides over the pixels: one thread per pixel would pass the workgroup limit of a large canvas.
    const uint pixels = frame.film.width * frame.film.height;
    for (uint pixel = id.x; pixel < pixels; pixel += frame.queues.gridThreads)
        buildSampleList(frame, pixel);
}
