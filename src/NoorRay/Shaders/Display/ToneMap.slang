import NoorRHI;
__exported import Display.Records;

//! Khronos PBR Neutral, from the reference implementation of the Khronos 3D Commerce group.
float3 applyNeutral(float3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;
    const float x = min(color.r, min(color.g, color.b));
    const float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;
    const float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression)
        return color;
    const float d = 1.0 - startCompression;
    const float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;
    const float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return lerp(color, float3(newPeak, newPeak, newPeak), g);
}

//! AgX base with the sixth-order contrast fit of Benjamin Wrensch; the matrices are Blender's AgX inset and outset in linear Rec.709.
float3 applyAgx(float3 color) {
    const float3x3 inset = float3x3(0.842479062253094, 0.0784335999999992, 0.0792237451477643, 0.0423282422610123, 0.878468636469772, 0.0791661274605434,
                                    0.0423756549057051, 0.0784336, 0.879142973793104);
    const float3x3 outset = float3x3(1.19687900512017, -0.0980208811401368, -0.0990297440797205, -0.0528968517574562, 1.15190312990417, -0.0989611768448433,
                                     -0.0529716355144438, -0.0980434501171241, 1.15107367264116);
    const float minEv = -12.47393;
    const float maxEv = 4.026069;
    float3 x = mul(inset, max(color, float3(1e-10, 1e-10, 1e-10)));
    x = saturate((log2(x) - minEv) / (maxEv - minEv));
    const float3 x2 = x * x;
    const float3 x4 = x2 * x2;
    x = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
    x = mul(outset, x);
    return pow(max(x, float3(0.0, 0.0, 0.0)), float3(2.2, 2.2, 2.2));
}

float3 encodeSrgb(float3 value) {
    const float3 low = value * 12.92;
    const float3 high = 1.055 * pow(value, float3(1.0 / 2.4, 1.0 / 2.4, 1.0 / 2.4)) - 0.055;
    return select(value <= 0.0031308, low, high);
}

struct ToneMapVaryings {
    float4 position : SV_Position;
};

ToneMapVaryings coverDisplay(uint vertex) {
    const float2 corner = float2(float((vertex << 1u) & 2u), float(vertex & 2u));
    return {float4(corner * 2.0 - 1.0, 0.0, 1.0)};
}

float4 displayFilm(DisplayArguments arguments, uint2 pixel) {
    float3 radiance = max(arguments.source[pixel.y * arguments.width + pixel.x].xyz, float3(0.0, 0.0, 0.0));
    if (arguments.splatted != 0) {
        const float4 splat = arguments.splats.Load(int2(pixel));
        radiance = max(splat.rgb, float3(0.0, 0.0, 0.0)) + (1.0 - saturate(splat.a)) * radiance;
    }
    radiance *= exp2(arguments.exposure);
    float3 mapped = saturate(radiance);
    if (arguments.toneMap == toneMapNeutral)
        mapped = applyNeutral(radiance);
    else if (arguments.toneMap == toneMapAgx)
        mapped = applyAgx(radiance);
    return float4(encodeSrgb(saturate(mapped)), 1.0);
}
