import NoorRHI;
import Common.Math;
import Common.Frame;
import SceneRecords;

//! SG lighting for the light tree importance, ported from Tokuyoshi's VSGL (MIT) after Tokuyoshi et al. 2024, "Hierarchical Light Sampling".

static const float sgFloatMin = 1.175494351e-38;
static const float sgFloatMaxValue = 3.402823466e38;
static const float sgFloatEpsilon = 1.192092896e-7;

float sgMulSign(float x, float y) {
    return asfloat((asuint(y) & 0x80000000u) ^ asuint(x));
}

//! exp(x) - 1 without cancellation, as Higham's Section 1.14.1.
float sgExpm1(float x) {
    const float u = exp(x);
    if (u == 1.0)
        return x;
    const float y = u - 1.0;
    if (abs(x) < 1.0)
        return y * x / log(u);
    return y;
}

//! (exp(x) - 1) / x without cancellation.
float sgExpm1OverX(float x) {
    const float u = exp(x);
    if (u == 1.0)
        return 1.0;
    const float y = u - 1.0;
    if (abs(x) < 1.0)
        return y / log(u);
    return y / x;
}

//! A polynomial erf with a smaller error than Abramowitz and Stegun 7.1.26.
float sgErf(float x) {
    const float a = abs(x);
    if (a >= 4.0)
        return sgMulSign(1.0, x);
    if (a > 1.0) {
        const float p =
            ((((((2.58907676e-5 * a - 5.64874616e-4) * a + 5.66795561e-3) * a - 3.51759829e-2) * a + 1.54329389e-1) * a + 9.15674746e-1) * a + 1.628459513) * a;
        return sgMulSign(1.0 - exp2(-p), x);
    }
    const float x2 = x * x;
    return (((((-5.58853149e-4 * x2 + 4.90735564e-3) * x2 - 2.67030653e-2) * x2 + 1.12799220e-1) * x2 - 3.76123011e-1) * x2 + 1.128379121) * x;
}

float sgErfc(float x) {
    return 1.0 - sgErf(x);
}

//! The integral over the sphere of the SG exp(sharpness * (dot(o, axis) - 1)).
float sgIntegral(float sharpness) {
    return 4.0 * pi * sgExpm1OverX(-2.0 * sharpness);
}

struct SgLobe {
    float3 axis;
    float sharpness;
    float logAmplitude;
};

//! The product of two SGs, with the log amplitude in a form without cancellation.
SgLobe sgProduct(float3 axis1, float sharpness1, float3 axis2, float sharpness2) {
    const float3 axis = axis1 * sharpness1 + axis2 * sharpness2;
    const float sharpness = length(axis);
    const float3 d = axis1 - axis2;
    SgLobe lobe;
    lobe.axis = axis / max(sharpness, sgFloatMin);
    lobe.sharpness = sharpness;
    lobe.logAmplitude = -sharpness1 * sharpness2 * dot(d, d) / max(sharpness + sharpness1 + sharpness2, sgFloatMin);
    return lobe;
}

//! The interpolation factor of the hemispherical integral of an SG (Tokuyoshi 2022, Eq. 1).
float sgNormalizedHemisphericalIntegral(float cosine, float sharpness) {
    const float steepness = sharpness * sqrt((0.5 * sharpness + 0.6517328826907056) / ((sharpness + 1.3418280033141288) * sharpness + 7.221668779895671));
    return saturate(0.5 + 0.5 * (sgErf(steepness * clamp(cosine, -1.0, 1.0)) / sgErf(steepness)));
}

//! The hemispherical integral of a vMF about a pole at the given cosine from its axis (Tokuyoshi et al. 2024, supplement Listing 4).
float vmfHemisphericalIntegral(float cosine, float sharpness) {
    const float e = exp(-sharpness);
    return lerp(e, 1.0, sgNormalizedHemisphericalIntegral(cosine, sharpness)) / (e + 1.0);
}

//! Supplement Listing 5: the integral of an SG about the normal times the clamped cosine, over 2 pi.
float upperSgClampedCosineIntegralOverTwoPi(float s) {
    if (s <= 0.5)
        return (((((((-1.0 / 362880.0) * s + 1.0 / 40320.0) * s - 1.0 / 5040.0) * s + 1.0 / 720.0) * s - 1.0 / 120.0) * s + 1.0 / 24.0) * s - 1.0 / 6.0) * s
               + 0.5;
    return (1.0 - sgExpm1OverX(-s)) / s;
}

//! Supplement Listing 6: the same for an SG about the negated normal.
float lowerSgClampedCosineIntegralOverTwoPi(float s) {
    const float e = exp(-s);
    if (s <= 0.5)
        return e
               * (((((((((1.0 / 403200.0) * s - 1.0 / 45360.0) * s + 1.0 / 5760.0) * s - 1.0 / 840.0) * s + 1.0 / 144.0) * s - 1.0 / 30.0) * s + 1.0 / 8.0) * s
                   - 1.0 / 3.0)
                      * s
                  + 0.5);
    return e * (sgExpm1OverX(-s) - e) / s;
}

//! Supplement Listing 7: the product integral of an SG and the clamped cosine over pi, which is never zero (Eq. 7 of the paper).
float sgClampedCosineProductIntegralOverPi(float cosine, float sharpness) {
    const float t = sharpness
                    * sqrt(0.5 * ((sharpness + 2.736083161127256) * sharpness + 17.021297781741875)
                           / (((sharpness + 4.010082672851042) * sharpness + 15.21915626314721) * sharpness + 76.08789627236074));
    const float tz = t * cosine;
    const float lerpFactor =
        saturate(max(0.5 * (cosine * sgErfc(-tz) + sgErfc(t)) - 0.5 * 0.5641895835477563 * exp(-tz * tz) * sgExpm1(t * t * (cosine * cosine - 1.0)) / t,
                     0.5 * sgFloatEpsilon));
    return 2.0 * lerp(lowerSgClampedCosineIntegralOverTwoPi(sharpness), upperSgClampedCosineIntegralOverTwoPi(sharpness), lerpFactor);
}

//! The receiver for which the light tree picks lights: a surface with a diffuse and an isotropic GGX lobe, or a point in a medium.
struct LightReceiver {
    //! Zero in a medium, where the phase is taken as isotropic.
    float3 normal;
    float3 view;
    float alpha;
    //! The share of the glossy lobe; the rest is diffuse.
    float glossy;
    //! The diffuse share also takes light from behind the surface.
    bool transmits;
    //! The visibility cache cell plus one, or zero for none; visibility points to the words of the cell: the learned stats and the cut of the light tree.
    uint cell;
    Ptr<uint> visibility;
    //! Picks the lights by their power alone, without the SG lighting of the tree nodes, as the deep vertices of a path do.
    bool byPower;
};

static const uint lightReceiverMedium = 1u;
static const uint lightReceiverTransmits = 2u;
static const uint lightReceiverByPower = 4u;
static const uint lightReceiverCellShift = 3u;

LightReceiver mediumReceiver() {
    LightReceiver receiver;
    receiver.normal = float3(0.0, 0.0, 0.0);
    receiver.view = float3(0.0, 0.0, 1.0);
    receiver.alpha = 1.0;
    receiver.glossy = 0.0;
    receiver.transmits = false;
    receiver.cell = 0;
    receiver.visibility = Ptr<uint>(0);
    receiver.byPower = false;
    return receiver;
}

LightReceiver diffuseReceiver(float3 normal, float3 view) {
    LightReceiver receiver;
    receiver.normal = normal;
    receiver.view = view;
    receiver.alpha = 1.0;
    receiver.glossy = 0.0;
    receiver.transmits = false;
    receiver.cell = 0;
    receiver.visibility = Ptr<uint>(0);
    receiver.byPower = false;
    return receiver;
}

//! NEE and MIS must see the same receiver, so the path keeps it packed and both unpack the same bits.
uint4 packReceiver(LightReceiver receiver) {
    const uint byPower = receiver.byPower ? lightReceiverByPower : 0u;
    if (all(receiver.normal == 0.0))
        return uint4(0u, 0u, 0u, lightReceiverMedium | byPower);
    return uint4(packOctahedral(receiver.normal), packOctahedral(receiver.view), f32tof16(receiver.alpha) | f32tof16(saturate(receiver.glossy)) << 16,
                 (receiver.transmits ? lightReceiverTransmits : 0u) | byPower | receiver.cell << lightReceiverCellShift);
}

//! visibility is the cells of the visibility cache, which a packed cell indexes.
LightReceiver unpackReceiver(uint4 packed, Ptr<uint> visibility) {
    if ((packed.w & lightReceiverMedium) != 0) {
        LightReceiver medium = mediumReceiver();
        medium.byPower = (packed.w & lightReceiverByPower) != 0;
        return medium;
    }
    LightReceiver receiver;
    receiver.normal = unpackOctahedral(packed.x);
    receiver.view = unpackOctahedral(packed.y);
    receiver.alpha = f16tof32(packed.z & 0xffffu);
    receiver.glossy = f16tof32(packed.z >> 16);
    receiver.transmits = (packed.w & lightReceiverTransmits) != 0;
    receiver.cell = packed.w >> lightReceiverCellShift;
    receiver.visibility = receiver.cell == 0 ? Ptr<uint>(0) : visibility + (receiver.cell - 1) * visibilityCellWords;
    receiver.byPower = (packed.w & lightReceiverByPower) != 0;
    return receiver;
}

//! The terms of a receiver that do not depend on the light, computed once per tree walk.
struct ReceiverLobes {
    bool medium;
    bool transmits;
    float diffuse;
    float glossy;
    float3 normal;
    Frame frame;
    float3 wi;
    float projectedAlpha2;
    //! JJ^T at the halfvector n, as (11, 12, 22), and 4 det(JJ^T).
    float3 jj;
    float detJJ4;
    //! The dominant reflection direction times the conservative sharpness of the reflection lobe, for the filtered visibility.
    float3 reflection;
    //! The words of the visibility cache cell of the receiver, or null.
    Ptr<uint> visibility;
    bool byPower;
};

ReceiverLobes prepareReceiver(LightReceiver receiver) {
    ReceiverLobes lobes;
    lobes.medium = all(receiver.normal == 0.0);
    lobes.transmits = receiver.transmits;
    lobes.visibility = receiver.visibility;
    lobes.byPower = receiver.byPower;
    lobes.normal = receiver.normal;
    lobes.frame = makeFrame(lobes.medium ? float3(0.0, 0.0, 1.0) : receiver.normal);
    lobes.wi = lobes.frame.toLocal(receiver.view);
    //! The glossy lobe reflects only a view above the normal; below it the receiver is diffuse.
    lobes.glossy = lobes.medium || lobes.wi.z <= 1e-4 ? 0.0 : saturate(receiver.glossy);
    lobes.diffuse = 1.0 - lobes.glossy;
    const float alpha2 = clamp(receiver.alpha * receiver.alpha, 1e-12, 1.0);
    lobes.projectedAlpha2 = alpha2 / max(1.0 - alpha2, sgFloatMin);
    const float wz = max(lobes.wi.z, 1e-4);
    const float vlen = length(lobes.wi.xy);
    const float2 v = vlen != 0.0 ? lobes.wi.xy / vlen : float2(1.0, 0.0);
    //! J = R(v) diag(0.5, 0.5 / wz), so JJ^T = R diag(0.25, 0.25 / wz^2) R^T.
    const float a = 0.25;
    const float b = 0.25 / (wz * wz);
    lobes.jj = float3(a * v.x * v.x + b * v.y * v.y, (a - b) * v.x * v.y, a * v.y * v.y + b * v.x * v.x);
    lobes.detJJ4 = 1.0 / (4.0 * wz * wz);
    const float reflectionSharpness = min((1.0 - alpha2) / max(2.0 * alpha2, sgFloatMin), lightSharpnessMax);
    const float3 dominant = normalize(float3(alpha2 * lobes.wi.xy, lobes.wi.z + sqrt(alpha2 * dot(lobes.wi.xy, lobes.wi.xy) + lobes.wi.z * lobes.wi.z)));
    lobes.reflection = lobes.frame.toWorld(reflect(-lobes.wi, dominant)) * reflectionSharpness;
    return lobes;
}

//! The symmetric GGX with the 2x2 roughness matrix (11, 12, 22).
float symmetricGgx(float3 m, float3 roughness) {
    const float det = max(roughness.x * roughness.z - roughness.y * roughness.y, sgFloatMin);
    const float2 adjugate = float2(roughness.z * m.x - roughness.y * m.y, -roughness.y * m.x + roughness.x * m.y);
    const float length2 = dot(m.xy, adjugate) / det + m.z * m.z;
    return 1.0 / (pi * sqrt(det) * length2 * length2);
}

//! The integral of an SG light of peak 1 against the BSDF lobes and cosine of the receiver: diffuse by Eq. 7, glossy by Eq. 12 with NDF filtering.
float receiverResponse(ReceiverLobes lobes, float3 axis, float sharpness) {
    if (lobes.medium)
        return sgIntegral(sharpness) / (4.0 * pi);
    const float cosine = clamp(dot(axis, lobes.normal), -1.0, 1.0);
    float diffuse = sgClampedCosineProductIntegralOverPi(cosine, sharpness);
    if (lobes.transmits)
        diffuse += sgClampedCosineProductIntegralOverPi(-cosine, sharpness);
    if (lobes.glossy <= 0.0)
        return diffuse;
    const float variance = 1.0 / sharpness;
    const float p = lobes.projectedAlpha2;
    const float3 filtered = float3(p + 2.0 * variance * lobes.jj.x, 2.0 * variance * lobes.jj.y, p + 2.0 * variance * lobes.jj.z);
    const float det = p * p + 2.0 * variance * (p * lobes.jj.x + p * lobes.jj.z) + variance * variance * lobes.detJJ4;
    const float denominator = 1.0 + filtered.x + filtered.z + det;
    float3 roughness;
    if (isinf(denominator) || isnan(denominator))
        roughness = float3(min(filtered.x, sgFloatMaxValue) / min(filtered.x + 1.0, sgFloatMaxValue), 0.0,
                           min(filtered.z, sgFloatMaxValue) / min(filtered.z + 1.0, sgFloatMaxValue));
    else
        roughness = min(filtered + float3(det, 0.0, det), sgFloatMaxValue) / denominator;
    const float3 halfway = lobes.wi + lobes.frame.toLocal(axis);
    const float halfwayLength = length(halfway);
    const float3 h = halfwayLength > 0.0 ? halfway / halfwayLength : float3(0.0, 0.0, 1.0);
    const float2 projected = float2(roughness.x * lobes.wi.x + roughness.y * lobes.wi.y, roughness.y * lobes.wi.x + roughness.z * lobes.wi.y);
    const float lobe = symmetricGgx(h, roughness) / (4.0 * sqrt(dot(lobes.wi.xy, projected) + lobes.wi.z * lobes.wi.z));
    const float3 product = lobes.reflection + axis * sharpness;
    const float productSharpness = length(product);
    const float visibility = vmfHemisphericalIntegral(dot(product / max(productSharpness, sgFloatMin), lobes.normal), productSharpness);
    return lobes.diffuse * diffuse + lobes.glossy * visibility * lobe * sgIntegral(sharpness);
}

//! The response of the receiver to a uniform environment of radiance 1, in the units of receiverResponse times the SG peak.
float uniformResponse(ReceiverLobes lobes) {
    if (lobes.medium)
        return 1.0;
    return lobes.diffuse * (lobes.transmits ? 2.0 : 1.0) + lobes.glossy;
}
