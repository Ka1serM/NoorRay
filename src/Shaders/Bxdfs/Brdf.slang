// BRDF and lighting functions
#ifndef BRDF_SLANG
#define BRDF_SLANG

#include "../Shared.h"
#include "../Bindings.slang"
#include "../Core/Core.slang"

float2 concentricMapping(float2 u);

// Fast normalization with epsilon guard - must be defined early as it's used by other functions
float3 fastNormalize(float3 v) {
    return v * rsqrt(max(dot(v, v), EPSILON));
}

// Coordinate system helpers
void buildCoordinateSystem(float3 N, out float3 T, out float3 B) {
    float3 n = fastNormalize(N);
    if (abs(n.z) < 0.999)
        T = fastNormalize(cross(n, float3(0.0, 0.0, 1.0)));
    else
        T = fastNormalize(cross(n, float3(0.0, 1.0, 0.0)));
    B = fastNormalize(cross(T, n));
}

float3 worldToLocal(float3 v, float3 T, float3 B, float3 N) {
    return float3(dot(v, T), dot(v, B), dot(v, N));
}

float3 localToWorld(float3 v, float3 T, float3 B, float3 N) {
    return T * v.x + B * v.y + N * v.z;
}

// Normal mapping helpers
static const bool NORMAL_MAP_IS_DIRECTX = true;
static const bool NORMAL_MAP_GENERAL_FLIP = true;

float3 decodeTangentNormal(float3 packedNormal) {
    float2 xy = packedNormal.xy * 2.0 - 1.0;
    if (NORMAL_MAP_IS_DIRECTX)
        xy.y = -xy.y;
    if (NORMAL_MAP_GENERAL_FLIP)
        xy = -xy;

    float z2 = max(1.0 - dot(xy, xy), 0.0);
    return float3(xy, sqrt(z2));
}

float3 applyNormalMap(float3 baseNormal, float3 interpolatedTangent, float tangentSign, float3 packedNormal) {
    float3 N = fastNormalize(baseNormal);
    float3 T = interpolatedTangent - N * dot(interpolatedTangent, N);
    float tLenSq = dot(T, T);
    float signValue = (abs(tangentSign) <= EPSILON) ? 1.0 : sign(tangentSign);
    float3 B;
    if (tLenSq <= EPSILON) {
        buildCoordinateSystem(N, T, B);
        B *= signValue;
    } else {
        T = T * rsqrt(tLenSq);
        B = fastNormalize(cross(N, T)) * signValue;
    }

    float3 tangentNormal = decodeTangentNormal(packedNormal);
    float3 mappedNormal = fastNormalize(mul(float3x3(T, B, N), tangentNormal));

    if (dot(mappedNormal, N) < 0.0)
        mappedNormal = -mappedNormal;

    return mappedNormal;
}

float3 stabilizeShadingNormal(float3 shadingNormal, float3 geometricNormal) {
    float3 Ns = fastNormalize(shadingNormal);
    float3 Ng = fastNormalize(geometricNormal);

    if (dot(Ns, Ng) < 0.0)
        Ns = -Ns;

    float alignment = clamp(dot(Ns, Ng), 0.0, 1.0);
    float keepShading = smoothstep(0.2, 0.8, alignment);
    return fastNormalize(lerp(Ng, Ns, keepShading));
}

float3 adaptShadingNormalLocal(float3 viewDir, float3 shadingNormal, float3 geometricNormal) {
    float3 Ns = fastNormalize(shadingNormal);
    float3 Ng = fastNormalize(geometricNormal);
    if (dot(Ns, Ng) < 0.0)
        Ns = -Ns;

    float3 wi = -viewDir;
    float3 r = reflect(wi, Ns);
    if (dot(r, Ng) < 0.0) {
        float3 t = r - Ng * dot(Ng, r);
        float tLenSq = dot(t, t);
        if (tLenSq > EPSILON)
            Ns = fastNormalize(-wi + fastNormalize(t));
        else
            Ns = Ng;
    }

    if (dot(Ns, Ng) < 0.0)
        Ns = -Ns;
    return Ns;
}

// Diffuse sampling
float3 sampleDiffuse(float3 N, inout uint rngState) {
    float2 d = concentricMapping(float2(rand(rngState), rand(rngState)));
    float z = sqrt(max(0.0, 1.0 - dot(d, d)));
    float3 local = float3(d, z);
    float3 T, B;
    buildCoordinateSystem(N, T, B);
    return fastNormalize(T * local.x + B * local.y + N * local.z);
}

float3 sampleUniformHemisphere(float3 N, inout uint rngState, out float pdf) {
    float u1 = rand(rngState);
    float u2 = rand(rngState);
    float z = u1;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * PI * u2;
    float3 local = float3(r * cos(phi), r * sin(phi), z);
    float3 T, B;
    buildCoordinateSystem(N, T, B);
    pdf = 0.5 / PI;
    return fastNormalize(T * local.x + B * local.y + N * local.z);
}

float regularizeSecondaryRoughness(float roughness, uint depth) {
    if (depth == 0u)
        return roughness;
    float minimumRoughness = 0.08 + 0.02 * min(float(depth - 1u), 3.0);
    return max(roughness, minimumRoughness);
}

float dielectricF0FromSpecular(float specular) {
    return clamp(0.04 * specular, 0.0, 0.16);
}

// Schlick approximation — accurate for conductors (metals).
float3 fresnelConductor(float3 F0, float cosTheta) {
    float m  = clamp(1.0 - cosTheta, 0.0, 1.0);
    float m2 = m * m;
    float m5 = m2 * m2 * m;
    return F0 + (1.0 - F0) * m5;
}

// Exact unpolarised Fresnel for a dielectric interface.
// cosThetaI > 0 means the ray is entering the material (etaI = outside, etaT = inside).
float fresnelDielectric(float cosThetaI, float etaI, float etaT) {
    float eta        = (cosThetaI >= 0.0) ? (etaT / etaI) : (etaI / etaT);
    float eta2       = eta * eta;
    float absCos     = abs(cosThetaI);
    float sinTheta2  = max(1.0 - absCos * absCos, 0.0);
    float cos2        = sqrt(max(eta2 - sinTheta2, 0.0));
    float rs = (absCos - cos2) / max(absCos + cos2, EPSILON);
    float rp = (eta2 * absCos - cos2) / max(eta2 * absCos + cos2, EPSILON);
    return 0.5 * (rs * rs + rp * rp);
}

// Derive the relative IOR from a specular-encoded F0 value.
// F0 = ((eta-1)/(eta+1))^2  →  eta = (1+sqrt(F0))/(1-sqrt(F0))
float f0ToEta(float f0) {
    float sqrtF0 = sqrt(clamp(f0, 0.0, 1.0 - EPSILON));
    return (1.0 + sqrtF0) / max(1.0 - sqrtF0, EPSILON);
}

// Exact dielectric Fresnel using an F0 derived from the specular parameter.
float fresnelDielectricFromSpecular(float specular, float cosTheta) {
    float f0  = dielectricF0FromSpecular(specular);
    float eta = f0ToEta(f0);
    return fresnelDielectric(cosTheta, 1.0, eta);
}

// Blended Fresnel: exact dielectric for non-metals, Schlick for conductors.
float3 fresnelMaterial(float3 F0, float metallic, float cosTheta, float specular) {
    float     dielectricF = fresnelDielectricFromSpecular(specular, cosTheta);
    float3    conductorF  = fresnelConductor(F0, cosTheta);
    return lerp(float3(dielectricF), conductorF, metallic);
}

float3 envBrdfApprox(float3 specularColor, float roughness, float nDotV) {
    float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * nDotV)) * r.x + r.y;
    float2 ab = float2(-1.04, 1.04) * a004 + r.zw;
    return specularColor * ab.x + ab.y;
}

float computeSpecularOcclusion(float nDotV, float ao, float roughness) {
    float clampedAo = saturate(ao);
    float clampedNdotV = saturate(nDotV);
    float clampedRoughness = saturate(roughness);
    return clamp(pow(max(clampedNdotV + clampedAo, EPSILON), exp2(-16.0 * clampedRoughness - 1.0)) - 1.0 + clampedAo, 0.0, 1.0);
}

float pdfDiffuse(float3 N, float3 L) {
    return max(dot(N, L), 0.0) / max(PI, EPSILON);
}

float2 concentricMapping(float2 u) {
    float2 uv = 2.0 * u - 1.0;
    if (abs(uv.x) <= EPSILON && abs(uv.y) <= EPSILON)
        return float2(0.0);

    float r;
    float theta;
    if (abs(uv.x) > abs(uv.y)) {
        r = uv.x;
        theta = 0.25 * PI * (uv.y / uv.x);
    } else {
        r = uv.y;
        theta = 0.5 * PI - 0.25 * PI * (uv.x / uv.y);
    }

    return r * float2(cos(theta), sin(theta));
}

float ggxAlpha2(float roughness) {
    float alpha = max(roughness * roughness, 0.001);
    return alpha * alpha;
}

float3 sampleVisibleHalfVectorGGX(float3 V, float3 N, float roughness, inout uint rngState) {
    float alpha = max(roughness * roughness, 0.001);
    float3 T, B;
    buildCoordinateSystem(N, T, B);

    float3 Vlocal = fastNormalize(worldToLocal(V, T, B, N));
    float3 Vh = fastNormalize(float3(alpha * Vlocal.x, alpha * Vlocal.y, Vlocal.z));

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > EPSILON ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(Vh, T1);

    float2 u = float2(rand(rngState), rand(rngState));
    float2 t = concentricMapping(u);
    float t1 = t.x;
    float t2 = t.y;
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;

    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    float3 Hlocal = fastNormalize(float3(alpha * Nh.x, alpha * Nh.y, max(Nh.z, 0.0)));
    return fastNormalize(localToWorld(Hlocal, T, B, N));
}

float distributionGGX(float3 N, float3 H, float roughness) {
    float a2 = ggxAlpha2(roughness);
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, EPSILON);
}

float ggxSmithDevshPart(float NdotX2, float a2) {
    return sqrt(max(a2 + (1.0 - a2) * NdotX2, EPSILON));
}

float ggxSmithG1(float NdotX, float a2) {
    return (2.0 * NdotX) / max(NdotX + ggxSmithDevshPart(NdotX * NdotX, a2), EPSILON);
}

float ggxSmithCorrelatedVisibility(float NdotV, float NdotL, float a2) {
    float Vterm = NdotL * ggxSmithDevshPart(NdotV * NdotV, a2);
    float Lterm = NdotV * ggxSmithDevshPart(NdotL * NdotL, a2);
    return 0.5 / max(Vterm + Lterm, EPSILON);
}

float3 evaluateSpecularBRDF(float3 normal, float3 viewDir, float3 sampledDir, float3 F, float roughness, float3 H) {
    float NdotV = max(dot(normal, viewDir), 0.0);
    float NdotL = max(dot(normal, sampledDir), 0.0);
    if (NdotV <= 0.0 || NdotL <= 0.0)
        return float3(0.0);

    float a2 = ggxAlpha2(roughness);
    float D = distributionGGX(normal, H, roughness);
    float Vis = ggxSmithCorrelatedVisibility(NdotV, NdotL, a2);
    return D * Vis * F;
}

float pdfSpecular(float3 V, float3 N, float3 H, float roughness) {
    float NdotV = max(dot(N, V), EPSILON);
    float a2 = ggxAlpha2(roughness);
    float D = distributionGGX(N, H, roughness);
    float G1 = ggxSmithG1(NdotV, a2);
    return D * G1 / max(4.0 * NdotV, EPSILON);
}

float computeSpecularSamplingWeight(float3 viewDir, float3 normal, float3 albedo, float3 F0, float metallic, float roughness, float specular) {
    float NdotV = max(dot(normal, viewDir), 0.0);
    float3 F = fresnelMaterial(F0, metallic, NdotV, specular);
    float weightDiffuse  = luminance((1.0 - F) * albedo) * (1.0 - metallic);
    float weightSpecular = luminance(F) + (1.0 - roughness) * 0.1;
    return clamp(weightSpecular / max(weightDiffuse + weightSpecular, EPSILON), 0.0, 1.0);
}

float orenNayarRecPiFactoredOut(float roughnessSquared, float VdotL, float NdotL, float NdotV) {
    float a2 = roughnessSquared * 0.5;
    float2 AB = float2(1.0, 0.0) + float2(-0.5, 0.45) * (float2(a2) / (float2(a2) + float2(0.33, 0.09)));
    float C = 1.0 / max(max(NdotL, NdotV), EPSILON);
    float cosPhiSinTheta = max(VdotL - NdotL * NdotV, 0.0);
    return AB.x + AB.y * cosPhiSinTheta * C;
}

float3 evaluateDiffuseBRDF(float3 albedo, float metallic, float roughness, float NdotL, float NdotV, float VdotL) {
    float clampedNdotL = max(NdotL, 0.0);
    float clampedNdotV = max(NdotV, 0.0);
    float roughnessSquared = roughness * roughness;
    float diffuse = orenNayarRecPiFactoredOut(roughnessSquared, VdotL, clampedNdotL, clampedNdotV) / PI;
    return (1.0 - metallic) * (albedo * diffuse);
}

#endif // BRDF_SLANG
