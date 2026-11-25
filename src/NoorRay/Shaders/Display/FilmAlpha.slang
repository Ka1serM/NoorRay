import NoorRHI;

//! The alpha of the film at a pixel of an upscaled image, filtered between the film pixels.
float filmAlpha(Ptr<float4> film, uint2 filmSize, uint2 pixel, uint2 size) {
    const float2 position = (float2(pixel) + 0.5) * float2(filmSize) / float2(size) - 0.5;
    const int2 base = int2(floor(position));
    const float2 weight = position - float2(base);
    const int2 last = int2(filmSize) - 1;
    const uint2 low = uint2(clamp(base, int2(0, 0), last));
    const uint2 high = uint2(clamp(base + 1, int2(0, 0), last));
    const float top = lerp(film[low.y * filmSize.x + low.x].w, film[low.y * filmSize.x + high.x].w, weight.x);
    const float bottom = lerp(film[high.y * filmSize.x + low.x].w, film[high.y * filmSize.x + high.x].w, weight.x);
    return lerp(top, bottom, weight.y);
}
