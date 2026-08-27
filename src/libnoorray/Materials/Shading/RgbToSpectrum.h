// Jakob & Hanika (2019) RGB-to-spectrum upsampling.
// Precomputed 64^3 sRGB table (9 MB) lives on the GPU; we trilinearly
// interpolate coefficients (c0,c1,c2) and evaluate the sigmoid polynomial
//   f(λ) = s(c0·λ² + c1·λ + c2)   with   s(x) = 0.5 + x / (2√(1+x²))
// Table layout identical to pbrt-v4 RGBToSpectrumTable / rgbspectrum_srgb.cpp.
#pragma once

#include <glm/vec3.hpp>

#include "Backend/Host/Platform.h"
#include "Materials/Shading/Spectrum.h"

// Table constants (matches pbrt-v4 RGBToSpectrumTable::res).
static constexpr int NrRgbSpecRes = 64;

// ── sigmoid s(x) = 0.5 + x/(2√(1+x²)) ────────────────────────────────────
NR_CPU_GPU inline float rgbSigmoid(float x)
{
    // Avoid float32 overflow of x*x for very large |x| (e.g. c2 = ±1e30f from
    // the uniform-RGB special-case in rgbLookupCoeffs).  For |x| > 1e10 the
    // sigmoid is 1 (resp. 0) to machine precision.
    if (x > 1e10f) return 1.f;
    if (x < -1e10f) return 0.f;
#if defined(NR_GPU_DEVICE_COMPILE)
    return 0.5f + x * rsqrtf(4.f + 4.f * x * x);
#else
    return 0.5f + x / (2.f * std::sqrt(1.f + x * x));
#endif
}

// Horner evaluation of c0·λ² + c1·λ + c2, then sigmoid.
NR_CPU_GPU inline float rgbSigmoidEval(float c0, float c1, float c2, float lambda)
{
    return rgbSigmoid(c2 + lambda * (c1 + lambda * c0));
}

// Binary search in a sorted 64-element array, returns largest i s.t. scale[i] < z.
// Result clamped to [0, NrRgbSpecRes-2].
NR_CPU_GPU inline int rgbFindInterval(const float* __restrict__ scale, float z)
{
    int lo = 0, hi = NrRgbSpecRes - 2;
    while (lo < hi)
    {
        const int mid = (lo + hi + 1) >> 1;
        if (scale[mid] < z) lo = mid;
        else                hi = mid - 1;
    }
    return lo;
}

// ── trilinear table lookup → (c0, c1, c2) ─────────────────────────────────
// scale : 64 floats   (z-node positions, non-uniform)
// coeffs: flat float[3][64][64][64][3]
NR_CPU_GPU inline void rgbLookupCoeffs(
    glm::vec3       rgb,
    const float* __restrict__ scale,
    const float* __restrict__ coeffs,
    float& c0, float& c1, float& c2)
{
    // Clamp to [0,1] — caller's responsibility for albedo; illuminant path already pre-scaled.
    rgb.x = fminf(fmaxf(rgb.x, 0.f), 1.f);
    rgb.y = fminf(fmaxf(rgb.y, 0.f), 1.f);
    rgb.z = fminf(fmaxf(rgb.z, 0.f), 1.f);

    // Special case: uniform rgb → closed-form achromatic polynomial
    if (rgb.x == rgb.y && rgb.y == rgb.z)
    {
        const float v = rgb.x;
        const float denom = v * (1.f - v);
        c0 = c1 = 0.f;
        c2 = (denom < 1e-10f)
                ? (v <= 0.5f ? -1e30f : 1e30f)
                : (v - 0.5f) / sqrtf(denom);
        return;
    }

    // Dominant channel
    int maxc;
    if (rgb.x > rgb.y) maxc = (rgb.x > rgb.z) ? 0 : 2;
    else               maxc = (rgb.y > rgb.z) ? 1 : 2;

    // Do not index glm::vec3 through &rgb.x here. Some GPU ABIs may pass an aligned
    // vec3 with padding, so pointer arithmetic can read the wrong components.
    const float z = maxc == 0 ? rgb.x : maxc == 1 ? rgb.y : rgb.z;
    const float xComponent = maxc == 0 ? rgb.y : maxc == 1 ? rgb.z : rgb.x;
    const float yComponent = maxc == 0 ? rgb.z : maxc == 1 ? rgb.x : rgb.y;
    const float x = xComponent * (NrRgbSpecRes - 1) / z;
    const float y = yComponent * (NrRgbSpecRes - 1) / z;

    const int xi = (int)x < NrRgbSpecRes - 2 ? (int)x : NrRgbSpecRes - 2;
    const int yi = (int)y < NrRgbSpecRes - 2 ? (int)y : NrRgbSpecRes - 2;
    const int zi = rgbFindInterval(scale, z);

    const float dx = x - (float)xi;
    const float dy = y - (float)yi;
    const float dz = (z - scale[zi]) / (scale[zi + 1] - scale[zi]);

    // Strides: [maxc][zi][yi][xi][coeff_i]
    constexpr int S3 = NrRgbSpecRes * NrRgbSpecRes * NrRgbSpecRes * 3;
    constexpr int S2 = NrRgbSpecRes * NrRgbSpecRes * 3;
    constexpr int S1 = NrRgbSpecRes * 3;
    const float* base = coeffs + maxc * S3;

    auto co = [&](int ax, int ay, int az, int i) -> float {
        return base[(zi + az) * S2 + (yi + ay) * S1 + (xi + ax) * 3 + i];
    };
    auto lerp = [](float t, float a, float b) -> float { return a + t * (b - a); };

    c0 = lerp(dz, lerp(dy, lerp(dx, co(0,0,0,0), co(1,0,0,0)),
                            lerp(dx, co(0,1,0,0), co(1,1,0,0))),
                  lerp(dy, lerp(dx, co(0,0,1,0), co(1,0,1,0)),
                            lerp(dx, co(0,1,1,0), co(1,1,1,0))));
    c1 = lerp(dz, lerp(dy, lerp(dx, co(0,0,0,1), co(1,0,0,1)),
                            lerp(dx, co(0,1,0,1), co(1,1,0,1))),
                  lerp(dy, lerp(dx, co(0,0,1,1), co(1,0,1,1)),
                            lerp(dx, co(0,1,1,1), co(1,1,1,1))));
    c2 = lerp(dz, lerp(dy, lerp(dx, co(0,0,0,2), co(1,0,0,2)),
                            lerp(dx, co(0,1,0,2), co(1,1,0,2))),
                  lerp(dy, lerp(dx, co(0,0,1,2), co(1,0,1,2)),
                            lerp(dx, co(0,1,1,2), co(1,1,1,2))));
}

// ── Public conversion functions ────────────────────────────────────────────

// Reflectance spectrum for albedo (rgb in [0,1]).
// Evaluates sigmoid polynomial at each sampled wavelength.
NR_CPU_GPU inline SampledSpectrum rgbAlbedoToSpectrum(
    glm::vec3 rgb,
    const SampledWavelengths& wl,
    const float* __restrict__ scale,
    const float* __restrict__ coeffs)
{
    float c0, c1, c2;
    rgbLookupCoeffs(rgb, scale, coeffs, c0, c1, c2);
    SampledSpectrum s;
    for (int i = 0; i < NrSpectrumSamples; ++i)
        s.values[i] = rgbSigmoidEval(c0, c1, c2, wl.lambda[i]);
    return s;
}

// Illuminant spectrum for light radiance (rgb may be > 1).
// Extracts scale = 2 * max(rgb), normalises, looks up polynomial, multiplies back.
// The factor of 2 matches pbrt-v4 RGBUnboundedSpectrum (keeps normalised rgb ≤ 0.5
// so the polynomial stays in a smooth regime).
NR_CPU_GPU inline SampledSpectrum rgbIlluminantToSpectrum(
    glm::vec3 rgb,
    const SampledWavelengths& wl,
    const float* __restrict__ scale,
    const float* __restrict__ coeffs,
    const float* __restrict__ d65)
{
    const float maxComp = fmaxf(rgb.x, fmaxf(rgb.y, rgb.z));
    if (maxComp <= 0.f)
        return SampledSpectrum(0.f);
    const float sc = 2.f * maxComp;
    const glm::vec3 rgbNorm = rgb / sc;
    float c0, c1, c2;
    rgbLookupCoeffs(rgbNorm, scale, coeffs, c0, c1, c2);
    SampledSpectrum s;
    for (int i = 0; i < NrSpectrumSamples; ++i)
    {
        const float d65Offset = (wl.lambda[i] - 300.f) * 0.2f;
        const int d65Index = static_cast<int>(d65Offset) < NrD65Samples - 2
            ? static_cast<int>(d65Offset) : NrD65Samples - 2;
        const float d65T = d65Offset - static_cast<float>(d65Index);
        const float illuminant = (d65[d65Index] + d65T *
            (d65[d65Index + 1] - d65[d65Index])) * NrD65Normalization;
        s.values[i] = sc * rgbSigmoidEval(c0, c1, c2, wl.lambda[i]) * illuminant;
    }
    return s;
}
