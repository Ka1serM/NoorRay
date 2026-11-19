import NoorRHI;
import Common.Math;

//! Directional albedo of charlieSheen, integrated numerically: rows are roughness 0.07, 1/7 ... 1, columns cosine 0.02, 1/7 ... 1.
static const float sheenAlbedoTable[64] = {
    1.3236, 0.6764, 0.3395, 0.1643, 0.0720, 0.0264, 0.0067, 0.0004, 1.0148, 0.6214, 0.3853, 0.2373, 0.1393, 0.0744, 0.0327, 0.0080,
    0.8026, 0.5549, 0.3947, 0.2833, 0.1991, 0.1333, 0.0811, 0.0396, 0.7126, 0.5186, 0.3908, 0.2988, 0.2261, 0.1659, 0.1147, 0.0703,
    0.6617, 0.4958, 0.3860, 0.3059, 0.2413, 0.1864, 0.1382, 0.0948, 0.6287, 0.4801, 0.3819, 0.3098, 0.2511, 0.2006, 0.1554, 0.1140,
    0.6054, 0.4686, 0.3785, 0.3122, 0.2579, 0.2109, 0.1685, 0.1292, 0.5882, 0.4599, 0.3756, 0.3137, 0.2629, 0.2188, 0.1788, 0.1414
};

float sheenTable(uint row, uint column) { return sheenAlbedoTable[row * 8 + column]; }

float sheenAlbedo(float cosine, float roughness) {
    const float x = saturate(cosine) * 7.0;
    const float y = saturate(roughness) * 7.0;
    const uint x0 = min(uint(x), 6u);
    const uint y0 = min(uint(y), 6u);
    const float tx = x - float(x0);
    const float ty = y - float(y0);
    return lerp(lerp(sheenTable(y0, x0), sheenTable(y0, x0 + 1), tx), lerp(sheenTable(y0 + 1, x0), sheenTable(y0 + 1, x0 + 1), tx), ty);
}

//! Estevez and Kulla 2017 Charlie distribution with the Neubelt and Pettineo visibility; returns f * cos(i) in a local frame. It exceeds unit albedo at grazing views, so callers divide by max(sheenAlbedo, 1).
float charlieSheen(float3 o, float3 i, float roughness) {
    if (o.z <= 0.0 || i.z <= 0.0)
        return 0.0;
    const float3 h = normalize(o + i);
    const float inverse = 1.0 / max(roughness, 0.07);
    const float sinTheta = safeSqrt(1.0 - h.z * h.z);
    const float distribution = (2.0 + inverse) * pow(sinTheta, inverse) * invTwoPi;
    const float visibility = 1.0 / (4.0 * (i.z + o.z - i.z * o.z));
    return distribution * visibility * i.z;
}
