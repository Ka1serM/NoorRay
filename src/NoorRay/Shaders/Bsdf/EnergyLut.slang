import NoorRHI;
import Common.Math;

//! Port of the old EnergyLut tables; Gpu/EnergyTables.cpp packs them in this order as unorm16 pairs.
static const uint ggxRoughnessSize = 32;
static const uint ggxCosineSize = 32;
static const uint dielectricF0Size = 16;
static const uint dielectricRoughnessSize = 16;
static const uint dielectricCosineSize = 16;
static const uint glassIorSize = 32;
static const uint glassIorHalfSize = 16;
static const uint glassRoughnessSize = 16;
static const uint glassCosineSize = 16;
static const float maximumDielectricF0 = 0.08;
static const float maximumGlassZ = 0.7453559924999299;

static const uint ggxDirectionalOffset = 0;
static const uint ggxAverageOffset = ggxDirectionalOffset + ggxCosineSize * ggxRoughnessSize;
static const uint dielectricDirectionalOffset = ggxAverageOffset + ggxRoughnessSize;
static const uint dielectricAverageOffset = dielectricDirectionalOffset + dielectricCosineSize * dielectricRoughnessSize * dielectricF0Size;
static const uint glassDirectionalOffset = dielectricAverageOffset + dielectricRoughnessSize * dielectricF0Size;
static const uint glassAverageOffset = glassDirectionalOffset + glassCosineSize * glassRoughnessSize * glassIorSize;

struct EnergyLut {
    Ptr<uint> tables;

    float decode(uint element) {
        const uint word = tables[element >> 1];
        const uint value = (element & 1u) != 0 ? (word >> 16) : (word & 0xffffu);
        return float(value) / 65535.0;
    }

    float sample2d(uint base, uint width, uint height, float x, float y) {
        const float px = saturate(x) * float(width - 1);
        const float py = saturate(y) * float(height - 1);
        const uint x0 = uint(px);
        const uint y0 = uint(py);
        const uint x1 = min(x0 + 1, width - 1);
        const uint y1 = min(y0 + 1, height - 1);
        const float tx = px - float(x0);
        const float ty = py - float(y0);
        return lerp(lerp(decode(base + y0 * width + x0), decode(base + y0 * width + x1), tx),
                    lerp(decode(base + y1 * width + x0), decode(base + y1 * width + x1), tx), ty);
    }

    float sample3d(uint base, uint width, uint height, uint depth, float x, float y, float z) {
        const float pz = saturate(z) * float(depth - 1);
        const uint z0 = uint(pz);
        const uint z1 = min(z0 + 1, depth - 1);
        return lerp(sample2d(base + z0 * width * height, width, height, x, y), sample2d(base + z1 * width * height, width, height, x, y), pz - float(z0));
    }

    float ggxDirectional(float roughness, float cosine) {
        return sample2d(ggxDirectionalOffset, ggxCosineSize, ggxRoughnessSize, sqrt(saturate(cosine)), roughness);
    }

    float ggxAverage(float roughness) { return sample2d(ggxAverageOffset, ggxRoughnessSize, 1, roughness, 0.0); }

    float dielectricDirectional(float roughness, float cosine, float f0) {
        return sample3d(dielectricDirectionalOffset, dielectricCosineSize, dielectricRoughnessSize, dielectricF0Size, sqrt(saturate(cosine)), roughness,
                        sqrt(saturate(f0 / maximumDielectricF0)));
    }

    float glassCoordinate(float eta) {
        const bool exiting = eta < 1.0;
        const float ior = exiting ? 1.0 / max(eta, 1e-5) : eta;
        const float side = saturate(safeSqrt((ior - 1.0) / (ior + 1.0)) / maximumGlassZ);
        const float index = exiting ? float(glassIorHalfSize - 1) * (1.0 - side) : float(glassIorHalfSize) + float(glassIorHalfSize - 1) * side;
        return index / float(glassIorSize - 1);
    }

    float glassDirectional(float roughness, float cosine, float eta) {
        if (abs(eta - 1.0) < 1e-4)
            return 1.0;
        return sample3d(glassDirectionalOffset, glassCosineSize, glassRoughnessSize, glassIorSize, sqrt(saturate(cosine)), roughness, glassCoordinate(eta));
    }

    float glassAverage(float roughness, float eta) {
        if (abs(eta - 1.0) < 1e-4)
            return 1.0;
        return sample2d(glassAverageOffset, glassRoughnessSize, glassIorSize, roughness, glassCoordinate(eta));
    }
};

//! Kulla-Conty: the Fresnel-weighted energy that single-scattering GGX misses, as an isotropic lobe.
float multipleScatterFresnel(float averageFresnel, float averageAlbedo) {
    const float f = saturate(averageFresnel);
    const float e = saturate(averageAlbedo);
    return f * f * e / max(1.0 - f * (1.0 - e), 1e-6);
}

//! The Kulla-Conty lobe value (without Fresnel) for the two cosines.
float multipleScatterLobe(EnergyLut lut, float roughness, float cosO, float cosI) {
    const float average = lut.ggxAverage(roughness);
    const float missing = 1.0 - average;
    if (missing <= 1e-6)
        return 0.0;
    return (1.0 - lut.ggxDirectional(roughness, cosO)) * (1.0 - lut.ggxDirectional(roughness, cosI)) / (pi * missing);
}
