import NoorRHI;
import Common.Math;

//! Below this alpha a lobe is a mirror or a smooth refraction, handled as a delta distribution.
static const float ggxDeltaAlpha = 1e-3;

//! Anisotropic GGX in a local frame with the normal along +Z.
struct Ggx {
    float2 alpha;

    bool isDelta() { return max(alpha.x, alpha.y) <= ggxDeltaAlpha; }

    float distribution(float3 h) {
        if (h.z <= 0.0)
            return 0.0;
        const float3 scaled = float3(h.x / alpha.x, h.y / alpha.y, h.z);
        const float d = dot(scaled, scaled);
        return 1.0 / (pi * alpha.x * alpha.y * d * d);
    }

    float lambda(float3 w) {
        const float cosSquared = w.z * w.z;
        if (cosSquared <= 0.0)
            return floatMax;
        const float tanSquared = (square(alpha.x * w.x) + square(alpha.y * w.y)) / cosSquared;
        return 0.5 * (sqrt(1.0 + tanSquared) - 1.0);
    }

    float g1(float3 w) { return 1.0 / (1.0 + lambda(w)); }
    float g2(float3 wo, float3 wi) { return 1.0 / (1.0 + lambda(wo) + lambda(wi)); }

    //! The density of visible normals for a view above the surface.
    float visibleDistribution(float3 wo, float3 h) {
        if (wo.z <= 0.0)
            return 0.0;
        return g1(wo) * max(dot(wo, h), 0.0) * distribution(h) / wo.z;
    }

    //! Heitz 2018 with the spherical-cap form of Dupuy and Benyoub 2023; wo.z must be positive.
    float3 sampleVisibleNormal(float3 wo, float2 u) {
        const float3 stretched = normalize(float3(alpha.x * wo.x, alpha.y * wo.y, wo.z));
        const float phi = twoPi * u.x;
        const float z = (1.0 - u.y) * (1.0 + stretched.z) - stretched.z;
        const float sinTheta = safeSqrt(1.0 - z * z);
        const float3 c = float3(sinTheta * cos(phi), sinTheta * sin(phi), z);
        const float3 hStretched = c + stretched;
        return normalize(float3(alpha.x * hStretched.x, alpha.y * hStretched.y, max(hStretched.z, 0.0)));
    }

    //! The scalar roughness of the energy tables for an anisotropic alpha.
    float tableRoughness() { return sqrt(sqrt(alpha.x * alpha.y)); }
};

Ggx makeGgx(float roughness, float anisotropic) {
    const float alpha = max(square(saturate(roughness)), 1e-4);
    Ggx ggx;
    if (anisotropic > 0.0) {
        const float aspect = sqrt(1.0 - 0.9 * saturate(anisotropic));
        ggx.alpha = float2(alpha / aspect, alpha * aspect);
    } else
        ggx.alpha = float2(alpha, alpha);
    return ggx;
}

//! Cycles' ensure_valid_specular_reflection: turns the shading normal towards Ng, in their plane, until the mirror of the view clears a threshold.
float3 clampShadingNormal(float3 geometricNormal, float3 shadingNormal, float3 view) {
    const float3 reflected = reflect(-view, shadingNormal);
    const float iz = dot(view, geometricNormal);
    const float threshold = min(0.9 * iz, 0.01);
    if (dot(geometricNormal, reflected) >= threshold)
        return shadingNormal;
    const float3 side = shadingNormal - dot(shadingNormal, geometricNormal) * geometricNormal;
    const float sideLengthSquared = dot(side, side);
    const float3 x = sideLengthSquared > 1e-20 ? side * rsqrt(sideLengthSquared) : shadingNormal;
    const float ix = dot(view, x);
    const float a = ix * ix + iz * iz;
    const float b = 2.0 * (a + iz * threshold);
    const float c = (threshold + iz) * (threshold + iz);
    const float root = safeSqrt(b * b - 4.0 * a * c);
    //! The root that solves the equation before squaring: ix and the right side keep their sign.
    const float nz2 = 0.25 * (ix < 0.0 ? b + root : b - root) / max(a, 1e-20);
    return safeSqrt(1.0 - nz2) * x + safeSqrt(nz2) * geometricNormal;
}
