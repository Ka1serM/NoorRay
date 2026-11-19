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
    float4 accumulated = film.accumulated[pixel];
    if (film.accumulate == 0) {
        if (value.w > 0.0)
            accumulated = value;
    } else
        accumulated += value;
    film.accumulated[pixel] = accumulated;
    film.output[pixel] = accumulated.w > 0.0 ? float4(accumulated.xyz / accumulated.w, 1.0) : float4(0.0, 0.0, 0.0, 1.0);
}
