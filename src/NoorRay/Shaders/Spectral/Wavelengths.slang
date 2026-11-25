import NoorRHI;
import SceneRecords;
import Common.Math;

//! Four hero wavelengths: basis maps linear sRGB to each, film is the colour matching weight over pdf (read from the table when used, a null table gives the sRGB primaries), weights is (4, 0, 0, 0) after dispersion.
struct Wavelengths {
    float4 lambda;
    float4 weights;
    Ptr<SpectralEntry> table;
    //! The table entries of the four wavelengths, 16 bits each.
    uint2 entries;

    uint entry(uint i) { return ((i < 2 ? entries.x : entries.y) >> ((i & 1u) * 16u)) & 0xffffu; }

    float3 basis(uint i) {
        if (isNull(table))
            return float3(i == 0 ? 1.0 : 0.0, i == 1 ? 1.0 : 0.0, i == 2 ? 1.0 : 0.0);
        return table[entry(i)].basisAndLambda.xyz;
    }

    float3 film(uint i) {
        if (isNull(table))
            return basis(i);
        return table[entry(i)].filmAndPdf.xyz;
    }

    float4 upsample(float3 rgb) { return float4(dot(rgb, basis(0)), dot(rgb, basis(1)), dot(rgb, basis(2)), dot(rgb, basis(3))); }

    float3 toFilm(float4 value) {
        const float4 weighted = value * weights * 0.25;
        return film(0) * weighted.x + film(1) * weighted.y + film(2) * weighted.z + film(3) * weighted.w;
    }

    bool isSecondaryTerminated() { return weights.y == 0.0; }

    [mutating]
    void terminateSecondary() {
        if (!isSecondaryTerminated())
            weights = float4(4.0, 0.0, 0.0, 0.0);
    }
};

float wavelengthOffset(float u, uint index) { return frac(u + float(index) * 0.25); }

Wavelengths sampleWavelengths(Ptr<SpectralEntry> table, float u, bool secondaryTerminated) {
    Wavelengths wavelengths;
    wavelengths.table = table;
    wavelengths.entries = uint2(0, 0);
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        const uint index = min(uint(wavelengthOffset(u, i) * float(spectralTableSize)), spectralTableSize - 1);
        wavelengths.lambda[i] = table[index].basisAndLambda.w;
        if (i < 2)
            wavelengths.entries.x |= index << (i * 16u);
        else
            wavelengths.entries.y |= index << ((i - 2) * 16u);
    }
    wavelengths.weights = secondaryTerminated ? float4(4.0, 0.0, 0.0, 0.0) : float4(1.0, 1.0, 1.0, 1.0);
    return wavelengths;
}

//! The sRGB primaries in the first three channels: the spectral BSDF code then shades RGB. The lambdas are the dominant wavelengths of the primaries.
Wavelengths rgbWavelengths() {
    Wavelengths wavelengths;
    wavelengths.lambda = float4(610.0, 550.0, 465.0, 550.0);
    wavelengths.table = Ptr<SpectralEntry>(0);
    wavelengths.entries = uint2(0, 0);
    wavelengths.weights = float4(4.0, 4.0, 4.0, 0.0);
    return wavelengths;
}

//! The hero sample is stored as 16 bits in the path state.
uint packHero(float u) { return min(uint(u * 65536.0), 65535u); }
float unpackHero(uint packed) { return (float(packed) + 0.5) / 65536.0; }
