import NoorRHI;
import SceneRecords;
import Common.Math;

//! Four equally spaced hero samples. basis converts linear sRGB to the value at each wavelength; film is the colour matching weight over the pdf. weights becomes (4, 0, 0, 0) when a dispersive event ends the secondary wavelengths.
struct Wavelengths {
    float4 lambda;
    float3 basis[4];
    float3 film[4];
    float4 weights;

    float4 upsample(float3 rgb) { return float4(dot(rgb, basis[0]), dot(rgb, basis[1]), dot(rgb, basis[2]), dot(rgb, basis[3])); }

    float3 toFilm(float4 value) {
        const float4 weighted = value * weights * 0.25;
        return film[0] * weighted.x + film[1] * weighted.y + film[2] * weighted.z + film[3] * weighted.w;
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
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        const uint index = min(uint(wavelengthOffset(u, i) * float(spectralTableSize)), spectralTableSize - 1);
        const SpectralEntry entry = table[index];
        wavelengths.lambda[i] = entry.basisAndLambda.w;
        wavelengths.basis[i] = entry.basisAndLambda.xyz;
        wavelengths.film[i] = entry.filmAndPdf.xyz;
    }
    wavelengths.weights = secondaryTerminated ? float4(4.0, 0.0, 0.0, 0.0) : float4(1.0, 1.0, 1.0, 1.0);
    return wavelengths;
}

//! The sRGB primaries in the first three channels: the spectral BSDF code then shades RGB. The lambdas are the dominant wavelengths of the primaries.
Wavelengths rgbWavelengths() {
    Wavelengths wavelengths;
    wavelengths.lambda = float4(610.0, 550.0, 465.0, 550.0);
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        wavelengths.basis[i] = float3(i == 0 ? 1.0 : 0.0, i == 1 ? 1.0 : 0.0, i == 2 ? 1.0 : 0.0);
        wavelengths.film[i] = wavelengths.basis[i];
    }
    wavelengths.weights = float4(4.0, 4.0, 4.0, 0.0);
    return wavelengths;
}

//! The hero sample is stored as 16 bits in the path state.
uint packHero(float u) { return min(uint(u * 65536.0), 65535u); }
float unpackHero(uint packed) { return (float(packed) + 0.5) / 65536.0; }
