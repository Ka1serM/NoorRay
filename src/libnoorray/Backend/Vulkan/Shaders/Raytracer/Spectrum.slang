#pragma once

// Spectral radiometry for the Vulkan path-tracing path.
//
// A 1:1 port of Materials/Shading/Spectrum.h (itself adapted from pbrt-v4).
// NrSpectrumSamples is 4, so SampledSpectrum maps exactly onto float4 and the
// arithmetic operators come for free; every constant and polynomial below is
// carried over verbatim so the Vulkan and CPU paths agree numerically.

static const int NrSpectrumSamples = 4;
static const float NrLambdaMin = 360.0;
static const float NrLambdaMax = 830.0;
static const float NrCIE_Y_integral = 106.856895;
static const int NrCIESamples = 471;
static const int NrD65Samples = 107;
static const float NrD65Normalization = 0.01011224743;

typedef float4 SampledSpectrum;

float spectrumAverage(SampledSpectrum s)
{
    return (s.x + s.y + s.z + s.w) / float(NrSpectrumSamples);
}

float spectrumMaxComponent(SampledSpectrum s)
{
    return max(max(s.x, s.y), max(s.z, s.w));
}

float spectrumMaxAbsComponent(SampledSpectrum s)
{
    return spectrumMaxComponent(abs(s));
}

bool spectrumNonZero(SampledSpectrum s)
{
    return any(s != SampledSpectrum(0.0));
}

bool spectrumIsFinite(SampledSpectrum s)
{
    return !any(isnan(s)) && !any(isinf(s));
}

// Returns 0 where the denominator is 0 (pbrt-v4 SafeDiv).
SampledSpectrum safeDiv(SampledSpectrum a, SampledSpectrum b)
{
    return select(b != SampledSpectrum(0.0), a / b, SampledSpectrum(0.0));
}

SampledSpectrum clampZero(SampledSpectrum s)
{
    return max(s, SampledSpectrum(0.0));
}

SampledSpectrum clampIndirectContribution(SampledSpectrum contribution,
    float clampValue, uint depth)
{
    if (depth == 0u || !(clampValue > 0.0))
        return contribution;
    const float limit = clampValue * 3.0;
    const float sum = dot(abs(contribution), float4(1.0));
    return sum > limit ? contribution * (limit / sum) : contribution;
}

// ---------------------------------------------------------------------------
// SampledWavelengths
// ---------------------------------------------------------------------------

struct SampledWavelengths
{
    float4 lambda;
    float4 pdf;

    SampledSpectrum PDF() { return pdf; }

    // Kill secondary wavelengths after dispersive refraction to avoid bias.
    [mutating] void terminateSecondary()
    {
        if (secondaryTerminated())
            return;
        pdf.y = 0.0;
        pdf.z = 0.0;
        pdf.w = 0.0;
        pdf.x /= float(NrSpectrumSamples);
    }

    bool secondaryTerminated()
    {
        return pdf.y == 0.0 && pdf.z == 0.0 && pdf.w == 0.0;
    }
};

// pbrt-v4 util/sampling.h
float sampleVisibleWavelengths(float u)
{
    return 538.0 - 138.888889 * atanh(0.85691062 - 1.82750197 * u);
}

float visibleWavelengthsPDF(float lambda)
{
    if (lambda < 360.0 || lambda > 830.0)
        return 0.0;
    const float x = 0.0072 * (lambda - 538.0);
    const float ch = cosh(x);
    return 0.0039398042 / (ch * ch);
}

// Uniform stratified sampling across [LambdaMin, LambdaMax] with wraparound.
SampledWavelengths sampleWavelengthsUniform(float u)
{
    SampledWavelengths wl;
    const float delta = (NrLambdaMax - NrLambdaMin) / float(NrSpectrumSamples);
    float first = NrLambdaMin + u * (NrLambdaMax - NrLambdaMin);
    float4 lambda;
    lambda.x = first;
    [unroll] for (int i = 1; i < NrSpectrumSamples; ++i)
    {
        float previous = lambda[i - 1] + delta;
        if (previous > NrLambdaMax)
            previous = NrLambdaMin + (previous - NrLambdaMax);
        lambda[i] = previous;
    }
    wl.lambda = lambda;
    wl.pdf = float4(1.0 / (NrLambdaMax - NrLambdaMin));
    return wl;
}

// Stratified sampling biased toward the photopic-sensitivity peak (~555 nm).
SampledWavelengths sampleWavelengthsVisible(float u)
{
    SampledWavelengths wl;
    float4 lambda;
    float4 pdf;
    [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
    {
        float up = u + float(i) / float(NrSpectrumSamples);
        if (up > 1.0)
            up -= 1.0;
        lambda[i] = sampleVisibleWavelengths(up);
        pdf[i] = visibleWavelengthsPDF(lambda[i]);
    }
    wl.lambda = lambda;
    wl.pdf = pdf;
    return wl;
}

SampledWavelengths sampleWavelengthsVisibleSingle(float u)
{
    SampledWavelengths wl;
    wl.lambda = float4(sampleVisibleWavelengths(u), 0.0, 0.0, 0.0);
    wl.pdf = float4(
        visibleWavelengthsPDF(wl.lambda.x) / float(NrSpectrumSamples), 0.0, 0.0, 0.0);
    return wl;
}

// ---------------------------------------------------------------------------
// Spectral tables
//
// The renderer passes descriptor-backed buffers (SceneData::cieX/cieY/cieZ/d65). Here the
// same values live in one storage buffer laid out by VulkanSpectralTables:
//   [0, 471)          CIE X
//   [471, 942)        CIE Y
//   [942, 1413)       CIE Z
//   [1413, 1520)      CIE D65
// ---------------------------------------------------------------------------

static const uint SpectralTableCieXOffset = 0u;
static const uint SpectralTableCieYOffset = 471u;
static const uint SpectralTableCieZOffset = 942u;
static const uint SpectralTableD65Offset = 1413u;
static const uint SpectralTableFloatCount = 1520u;

// Table: 471 entries at 1 nm steps, offset = round(lambda) - 360.
float sampleCIETable(ByteAddressBuffer tables, uint tableOffset, float lambda)
{
    const int offset = int(lambda + 0.5) - 360;
    if (offset < 0 || offset >= NrCIESamples)
        return 0.0;
    return asfloat(tables.Load((tableOffset + uint(offset)) * 4u));
}

SampledSpectrum sampleCIESpectrum(ByteAddressBuffer tables, uint tableOffset,
    SampledWavelengths wl)
{
    SampledSpectrum result;
    [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
        result[i] = sampleCIETable(tables, tableOffset, wl.lambda[i]);
    return result;
}

// Spectral radiance -> XYZ via the Monte Carlo estimator (pbrt-v4 ToXYZ).
float3 spectrumToXYZ(SampledSpectrum L, SampledWavelengths wl, ByteAddressBuffer tables)
{
    const SampledSpectrum X = sampleCIESpectrum(tables, SpectralTableCieXOffset, wl);
    const SampledSpectrum Y = sampleCIESpectrum(tables, SpectralTableCieYOffset, wl);
    const SampledSpectrum Z = sampleCIESpectrum(tables, SpectralTableCieZOffset, wl);
    const SampledSpectrum pdf = wl.PDF();
    return float3(
        spectrumAverage(safeDiv(X * L, pdf)),
        spectrumAverage(safeDiv(Y * L, pdf)),
        spectrumAverage(safeDiv(Z * L, pdf))) / NrCIE_Y_integral;
}

// Y (luminance) only - for Russian roulette.
float spectrumY(SampledSpectrum L, SampledWavelengths wl, ByteAddressBuffer tables)
{
    const SampledSpectrum Y = sampleCIESpectrum(tables, SpectralTableCieYOffset, wl);
    return spectrumAverage(safeDiv(Y * L, wl.PDF())) / NrCIE_Y_integral;
}

// CIE D65, 300--830 nm at 5 nm, normalized as pbrt does.
float sampleD65(ByteAddressBuffer tables, float lambda)
{
    const float position = (lambda - 300.0) / 5.0;
    if (position < 0.0 || position > float(NrD65Samples - 1))
        return 0.0;
    const int index = int(position);
    const float fraction = position - float(index);
    const int next = min(index + 1, NrD65Samples - 1);
    const float a = asfloat(tables.Load((SpectralTableD65Offset + uint(index)) * 4u));
    const float b = asfloat(tables.Load((SpectralTableD65Offset + uint(next)) * 4u));
    return lerp(a, b, fraction) * NrD65Normalization;
}

SampledSpectrum sampleD65Spectrum(ByteAddressBuffer tables, SampledWavelengths wl)
{
    SampledSpectrum result;
    [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
        result[i] = sampleD65(tables, wl.lambda[i]);
    return result;
}

// ---------------------------------------------------------------------------
// Colour conversion
// ---------------------------------------------------------------------------

// XYZ -> linear sRGB (D65 whitepoint, ITU-R BT.709 primaries), IEC 61966-2-1.
float3 xyzToLinearSRGB(float3 xyz)
{
    return float3(
         3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z,
        -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z,
         0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z);
}

// Minimax polynomial approximation from pbrt-v4 util/color.h.
float linearToSRGB(float value)
{
    if (value <= 0.0031308)
        return 12.92 * value;
    const float sq = sqrt(max(0.0, value));
    const float p =
        -0.0016829072605308378 +
        sq * ( 0.03453868659826638 +
        sq * ( 0.7642611304733891 +
        sq * ( 2.0041169284241644 +
        sq * ( 0.7551545191665577 +
        sq * -0.016202083165206348))));
    const float q =
         4.178892964897981e-7 +
        sq * (-0.00004375359692957097 +
        sq * ( 0.03467195408529984 +
        sq * ( 0.6085338522168684 +
        sq * ( 1.8970238036421054 +
        sq *   1.0))));
    return (p / q) * value;
}
