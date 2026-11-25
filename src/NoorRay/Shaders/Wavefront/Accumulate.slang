import NoorRHI;
import Wavefront.Records;
import Common.Math;

//! Per pixel: adds the frame sums to the accumulation (or replaces it without accumulation) and writes the mean to output.
void accumulateFilm(WavefrontFrame frame, uint pixel) {
    const FilmRecord film = frame.film;
    if (pixel >= film.width * film.height)
        return;
    const AtomicPtr<float> sums = film.frame + pixel * 4;
    const float4 value = float4(atomicLoad(sums), atomicLoad(sums + 1), atomicLoad(sums + 2), atomicLoad(sums + 3));
    atomicStore(sums, 0.0);
    atomicStore(sums + 1, 0.0);
    atomicStore(sums + 2, 0.0);
    atomicStore(sums + 3, 0.0);
    const AtomicPtr<float> transparentSum = film.frame + film.width * film.height * 4 + pixel;
    const uint transparent = uint(atomicLoad(transparentSum));
    atomicStore(transparentSum, 0.0);
    float4 accumulated = film.accumulated[pixel];
    uint4 state = film.pixels[pixel];
    if (film.accumulate == 0) {
        if (value.w > 0.0) {
            accumulated = value;
            state.w = transparent;
        }
    } else {
        accumulated += value;
        state.w += transparent;
    }
    film.pixels[pixel] = state;
    film.accumulated[pixel] = accumulated;
    const float alpha = saturate(1.0 - float(state.w) / max(accumulated.w, 1.0));
    const float4 mean = accumulated.w > 0.0 ? float4(accumulated.xyz / accumulated.w * film.exposure, alpha) : float4(0.0, 0.0, 0.0, 1.0);
    film.output[pixel] = mean;
    if (film.shared.value != 0)
        film.shared[uint2(pixel % film.width, pixel / film.width)] = mean;
}
