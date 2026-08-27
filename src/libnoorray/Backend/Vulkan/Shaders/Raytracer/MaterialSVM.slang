#pragma once

#include "CompositeBsdf.slang"
#include "RgbToSpectrum.slang"
#include "Sellmeier.slang"

// Vulkan MaterialX SVM decoder.  The stream and stack encoding mirrors
// Materials/SVM/SvmTypes.h: node payloads are read as words, and linked
// values are NaN-tagged stack offsets.  Keeping decoding explicit avoids a
// Slang/C++ packing mismatch and lets this evaluator be shared by regular and
// Gaussian hit programs.

static const uint SVM_END = 0;
static const uint SVM_MATH = 3;
static const uint SVM_VECTOR_MATH = 4;
static const uint SVM_MIX = 5;
static const uint SVM_CLAMP = 6;
static const uint SVM_REMAP_RANGE = 7;
static const uint SVM_GAMMA = 9;
static const uint SVM_HSV_ADJUST = 10;
static const uint SVM_INVERT = 11;
static const uint SVM_CONTRAST = 12;
static const uint SVM_SATURATE = 13;
static const uint SVM_SEPARATE_COLOR = 15;
static const uint SVM_COMBINE_COLOR = 16;
static const uint SVM_TEX_COORD = 17;
static const uint SVM_IMAGE_TEXTURE = 20;
// Keep these values in lock-step with nr::svm::NodeType.  The value and
// conversion opcodes occupy 1 and 2 even though they normally fold away in
// the compiler; closure records therefore start at 30, not 31.
static const uint SVM_NOISE = 21;
static const uint SVM_FRACTAL_NOISE = 22;
static const uint SVM_CHECKER = 23;
static const uint SVM_GRADIENT = 24;
static const uint SVM_WORLEY = 25;
static const uint SVM_CELL = 26;
static const uint SVM_UNIFIED_NOISE = 27;
static const uint SVM_NORMAL_MAP = 28;
static const uint SVM_BUMP = 29;
static const uint SVM_CLOSURE_WEIGHT = 30;
static const uint SVM_CLOSURE_DIFFUSE = 31;
static const uint SVM_CLOSURE_CONDUCTOR = 32;
static const uint SVM_CLOSURE_DIELECTRIC = 33;
static const uint SVM_CLOSURE_SHEEN = 34;
static const uint SVM_CLOSURE_SUBSURFACE = 35;
static const uint SVM_CLOSURE_EDF = 36;
static const uint SVM_CLOSURE_OPEN_PBR = 37;
static const uint SVM_ARTISTIC_IOR = 38;
static const uint SVM_MIX_CLOSURE = 39;
static const uint SVM_ADD_CLOSURE = 40;
static const uint SVM_JUMP_IF_ZERO = 41;
static const uint SVM_JUMP_IF_ONE = 42;
static const uint SVM_SURFACE_OUTPUT = 43;
static const uint SVM_SEPARATE_COLOR4 = 44;
static const uint SVM_COMBINE_COLOR4 = 45;
static const uint SVM_PREMULTIPLY = 46;
static const uint SVM_UNPREMULTIPLY = 47;
static const uint SVM_ROTATE2D = 48;
static const uint SVM_TIME = 49;
static const uint SVM_MATRIX_VALUE = 50;
static const uint SVM_MATRIX_COMPOSE = 51;
static const uint SVM_TRANSFORM_MATRIX = 52;
static const uint SVM_TRANSFORM = 53;
static const uint SVM_RANGE4 = 54;
static const uint SVM_COMBINE_COLOR2 = 55;
static const uint SVM_MATRIX_BINARY = 56;
static const uint SVM_MATRIX_UNARY = 57;
static const uint SVM_MATRIX_SELECT = 58;
static const uint SVM_MATRIX_DETERMINANT = 59;
static const uint SVM_STACK_NAN_MASK = 0x7FC00000u;
static const uint SVM_INVALID_OFFSET = 255u;
// Keep the closest-hit local frame bounded. The compiler records each
// material's peak stack size; 64 covers the native/MaterialX fast path while
// invalid accesses are safely ignored by the guarded helpers below.
static const uint SVM_STACK_SIZE = 64u;

interface IMaterialSvmTextures
{
    float4 sample(uint slot, float2 uv, uint filterType, uint addressU,
        uint addressV);
};

struct NullMaterialSvmTextures : IMaterialSvmTextures
{
    float4 sample(uint slot, float2 uv, uint filterType, uint addressU,
        uint addressV) { return float4(1.0); }
};

// Texture indices in a compiled SVM program are scene-texture indices on the
// host, but are rewritten to sampled-image heap indices before upload.  This
// policy resolves those indices directly from the immutable material buffer;
// no descriptor-set or legacy sampler state is involved.
struct DescriptorMaterialSvmTextures : IMaterialSvmTextures
{
    ByteAddressBuffer textureTable;
    uint textureOffset;
    uint textureCount;

    float addressCoordinate(float value, uint mode, inout bool outside)
    {
        if (mode == 3u && (value < 0.0 || value > 1.0))
            outside = true;
        if (mode == 1u || mode == 3u)
            return clamp(value, 0.0, 1.0);
        if (mode == 2u)
        {
            const float period = floor(value);
            const float fraction = value - period;
            return (int(period) & 1) != 0 ? 1.0 - fraction : fraction;
        }
        return value - floor(value);
    }

    float4 sample(uint slot, float2 uv, uint filterType, uint addressU,
        uint addressV)
    {
        if (slot >= textureCount)
            return float4(1.0);
        const uint descriptorIndex = textureTable.Load(
            (textureOffset + slot) * 4u);
        if (descriptorIndex == 0xFFFFFFFFu)
            return float4(1.0);

        bool outside = false;
        uv.x = addressCoordinate(uv.x, addressU, outside);
        uv.y = addressCoordinate(uv.y, addressV, outside);
        if (outside)
            return float4(0.0);

        Texture2D<float4> image = DescriptorHandle<Texture2D<float4>>(
            uint2(descriptorIndex, 0));
        uint width = 0u;
        uint height = 0u;
        image.GetDimensions(width, height);
        if (width == 0u || height == 0u)
            return float4(1.0);

        const float2 size = float2(width, height);
        if (filterType != 0u)
        {
            const int2 pixel = int2(min(uint2(max(uv * size, 0.0)),
                uint2(width - 1u, height - 1u)));
            return image.Load(int3(pixel, 0));
        }

        // Match the host image sampler's normalized-coordinate bilinear
        // convention.  Clamping the four taps also handles the edge texels
        // without relying on an implicit sampler object.
        const float2 p = uv * size - 0.5;
        const int2 base = int2(floor(p));
        const float2 f = frac(p);
        const int2 maxPixel = int2(width - 1u, height - 1u);
        const int2 p00 = clamp(base, 0, maxPixel);
        const int2 p10 = clamp(base + int2(1, 0), 0, maxPixel);
        const int2 p01 = clamp(base + int2(0, 1), 0, maxPixel);
        const int2 p11 = clamp(base + int2(1, 1), 0, maxPixel);
        const float4 x0 = lerp(image.Load(int3(p00, 0)),
            image.Load(int3(p10, 0)), f.x);
        const float4 x1 = lerp(image.Load(int3(p01, 0)),
            image.Load(int3(p11, 0)), f.x);
        return lerp(x0, x1, f.y);
    }
};

struct MaterialSvmContext
{
    float3 position;
    float3 geometricNormal;
    float3 interpolatedNormal;
    float3 tangent;
    float3 bitangent;
    float2 uv;
    float4 vertexColor;
    float3 viewDirection;
    float time;
};

struct MaterialSvmResult
{
    float3 albedo;
    float3 emission;
    float emissionStrength;
    float opacity;
    float transmission;
    float3 shadingNormal;
    uint flags;
};

static const uint SVM_HAS_SHADING_NORMAL = 1u;
static const uint SVM_HAS_EMISSION = 2u;
static const uint SVM_HAS_CUTOUT = 4u;

uint svmWord(ByteAddressBuffer words, uint offset)
{ return words.Load(offset * 4u); }

bool svmStackWord(uint value)
{ return (value & SVM_STACK_NAN_MASK) == SVM_STACK_NAN_MASK; }

bool svmInvalidInput(uint value)
{ return value == SVM_INVALID_OFFSET || (svmStackWord(value) && (value & 255u) == SVM_INVALID_OFFSET); }

float svmValue(ByteAddressBuffer words, float stack[SVM_STACK_SIZE], uint value)
{
    if (!svmStackWord(value))
        return asfloat(value);
    const uint slot = value & 255u;
    return slot < SVM_STACK_SIZE ? stack[slot] : 0.0;
}

float3 svmColor(ByteAddressBuffer words, float stack[SVM_STACK_SIZE], uint x,
    uint y, uint z)
{ return float3(svmValue(words, stack, x), svmValue(words, stack, y), svmValue(words, stack, z)); }

void svmStoreScalar(inout float stack[SVM_STACK_SIZE], uint offset, float value)
{
    if (offset < SVM_STACK_SIZE)
        stack[offset] = value;
}

void svmStoreColor(inout float stack[SVM_STACK_SIZE], uint offset, float3 value)
{
    svmStoreScalar(stack, offset, value.x);
    svmStoreScalar(stack, offset + 1u, value.y);
    svmStoreScalar(stack, offset + 2u, value.z);
}

float3 svmRgbToHsv(float3 c)
{
    float4 k = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = c.y < c.z ? float4(c.z, c.y, k.w, k.z) : float4(c.y, c.z, k.x, k.y);
    float4 q = c.x < p.x ? float4(p.x, p.y, p.w, c.x) : float4(c.x, p.y, p.z, p.x);
    float d = q.x - min(q.w, q.y);
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + 1e-10)), d / (q.x + 1e-10), q.x);
}

float3 svmHsvToRgb(float3 c)
{
    float3 p = abs(frac(c.xxx + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * lerp(float3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

// MaterialX's noise nodes use the same integer-gradient construction as the
// native SVM. Keep the implementation local to the shader so procedural
// materials do not fall back to a constant when evaluated on Vulkan.
uint svmNoiseRotate(uint x, uint k)
{ return (x << k) | (x >> (32u - k)); }

uint svmNoiseHash3(int x, int y, int z)
{
    uint a = 0xdeadbeefu + (3u << 2u) + 13u + uint(x);
    uint b = 0xdeadbeefu + (3u << 2u) + 13u + uint(y);
    uint c = 0xdeadbeefu + (3u << 2u) + 13u + uint(z);
    c ^= b; c -= svmNoiseRotate(b, 14u); a ^= c; a -= svmNoiseRotate(c, 11u);
    b ^= a; b -= svmNoiseRotate(a, 25u); c ^= b; c -= svmNoiseRotate(b, 16u);
    a ^= c; a -= svmNoiseRotate(c, 4u); b ^= a; b -= svmNoiseRotate(a, 14u);
    c ^= b; c -= svmNoiseRotate(b, 24u);
    return c;
}

float svmNoiseFade(float t)
{ return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

float svmNoiseGradient(uint hash, float x, float y, float z)
{
    const uint h = hash & 15u;
    const float u = h < 8u ? x : y;
    const float v = h < 4u ? y : ((h == 12u || h == 14u) ? x : z);
    return ((h & 1u) != 0u ? -u : u) + ((h & 2u) != 0u ? -v : v);
}

float svmPerlin3d(float3 p)
{
    const float3 cell = floor(p);
    const int3 i = int3(cell);
    const float3 f = p - cell;
    const float u = svmNoiseFade(f.x), v = svmNoiseFade(f.y), w = svmNoiseFade(f.z);
    const float n000 = svmNoiseGradient(svmNoiseHash3(i.x, i.y, i.z), f.x, f.y, f.z);
    const float n100 = svmNoiseGradient(svmNoiseHash3(i.x + 1, i.y, i.z), f.x - 1.0, f.y, f.z);
    const float n010 = svmNoiseGradient(svmNoiseHash3(i.x, i.y + 1, i.z), f.x, f.y - 1.0, f.z);
    const float n110 = svmNoiseGradient(svmNoiseHash3(i.x + 1, i.y + 1, i.z), f.x - 1.0, f.y - 1.0, f.z);
    const float n001 = svmNoiseGradient(svmNoiseHash3(i.x, i.y, i.z + 1), f.x, f.y, f.z - 1.0);
    const float n101 = svmNoiseGradient(svmNoiseHash3(i.x + 1, i.y, i.z + 1), f.x - 1.0, f.y, f.z - 1.0);
    const float n011 = svmNoiseGradient(svmNoiseHash3(i.x, i.y + 1, i.z + 1), f.x, f.y - 1.0, f.z - 1.0);
    const float n111 = svmNoiseGradient(svmNoiseHash3(i.x + 1, i.y + 1, i.z + 1), f.x - 1.0, f.y - 1.0, f.z - 1.0);
    const float z0 = lerp(lerp(n000, n100, u), lerp(n010, n110, u), v);
    const float z1 = lerp(lerp(n001, n101, u), lerp(n011, n111, u), v);
    return 0.9820 * lerp(z0, z1, w);
}

float svmPerlin2d(float2 p)
{
    const float2 cell = floor(p);
    const int2 i = int2(cell);
    const float2 f = p - cell;
    const float u = svmNoiseFade(f.x), v = svmNoiseFade(f.y);
    const float n00 = svmNoiseGradient(svmNoiseHash3(i.x, i.y, 0), f.x, f.y, 0.0);
    const float n10 = svmNoiseGradient(svmNoiseHash3(i.x + 1, i.y, 0), f.x - 1.0, f.y, 0.0);
    const float n01 = svmNoiseGradient(svmNoiseHash3(i.x, i.y + 1, 0), f.x, f.y - 1.0, 0.0);
    const float n11 = svmNoiseGradient(svmNoiseHash3(i.x + 1, i.y + 1, 0), f.x - 1.0, f.y - 1.0, 0.0);
    return 0.6616 * lerp(lerp(n00, n10, u), lerp(n01, n11, u), v);
}

float svmFractal(float3 p, bool is2d, int octaves, float lacunarity, float diminish)
{
    float value = 0.0;
    float amplitude = 1.0;
    octaves = clamp(octaves, 0, 16);
    [loop]
    for (int i = 0; i < octaves; ++i)
    {
        value += amplitude * (is2d ? svmPerlin2d(p.xy) : svmPerlin3d(p));
        amplitude *= diminish;
        p *= lacunarity;
    }
    return value;
}

float svmCellNoise(float3 p, bool is2d)
{
    const int3 i = int3(floor(p));
    const uint h = svmNoiseHash3(i.x, i.y, is2d ? 0 : i.z);
    return float(h & 0x00ffffffu) / 16777216.0;
}

float svmWorleyNoise(float3 p, bool is2d)
{
    const int3 center = int3(floor(p));
    float nearest = 1e20;
    [loop]
    for (int z = -1; z <= 1; ++z)
    {
        if (is2d && z != 0) continue;
        [loop]
        for (int y = -1; y <= 1; ++y)
        {
            [loop]
            for (int x = -1; x <= 1; ++x)
            {
                const int3 cell = center + int3(x, y, z);
                const uint h = svmNoiseHash3(cell.x, cell.y, cell.z);
                const float3 feature = float3(
                    float(h & 255u) / 256.0,
                    float((h >> 8u) & 255u) / 256.0,
                    is2d ? 0.0 : float((h >> 16u) & 255u) / 256.0);
                const float3 delta = float3(cell) + feature - p;
                nearest = min(nearest, dot(delta, delta));
            }
        }
    }
    return sqrt(nearest);
}

float svmMath(uint op, float a, float b, float c)
{
    switch (op) {
    case 0: return a + b; case 1: return a - b; case 2: return a * b;
    case 3: return abs(b) > 1e-20 ? a / b : 0.0;
    case 4: return abs(b) > 1e-20 ? a - floor(a / b) * b : 0.0;
    case 5: return pow(abs(a), b) * (a < 0.0 && frac(b) != 0.0 ? -1.0 : 1.0);
    case 6: return abs(a); case 7: return floor(a); case 8: return ceil(a);
    case 9: return floor(a + 0.5); case 10: return a < 0.0 ? -1.0 : (a > 0.0 ? 1.0 : 0.0);
    case 11: return min(a, b); case 12: return max(a, b); case 13: return sin(a);
    case 14: return cos(a); case 15: return tan(a); case 16: return asin(clamp(a, -1.0, 1.0));
    case 17: return acos(clamp(a, -1.0, 1.0)); case 18: return atan2(a, b);
    case 19: return log(max(a, 1e-20)); case 20: return exp(a); case 21: return sqrt(max(a, 0.0));
    case 22: return rsqrt(max(a, 1e-20)); case 23: return frac(a);
    case 24: return a > b ? 1.0 : 0.0; case 25: return a >= b ? 1.0 : 0.0;
    case 26: return a == b ? 1.0 : 0.0; case 27: return a != 0.0 && b != 0.0 ? 1.0 : 0.0;
    case 28: return a != 0.0 || b != 0.0 ? 1.0 : 0.0; case 29: return a == 0.0 ? 1.0 : 0.0;
    default: return c;
    }
}

float3 svmMix(float3 a, float3 b, float f, uint blend)
{
    switch (blend) {
    case 1: return a + f * b; case 2: return a - f * b;
    case 3: return 1.0 - (1.0 - a) * (1.0 - f * b); case 4: return abs(a - f * b);
    case 5: return select(a < 0.5, 2.0 * a * f * b,
        1.0 - 2.0 * (1.0 - a) * (1.0 - f * b));
    case 6: return select(f * b < 1.0,
        a / max(1.0 - f * b, 1e-20), a);
    case 7: return select(f * b > 0.0,
        1.0 - (1.0 - a) / max(f * b, 1e-20), a);
    default: return lerp(a, b, f);
    }
}

// The closure sink is deliberately separate from bytecode decoding.  The
// regular and Gaussian integrators can provide different BSDF implementations
// while retaining one material ABI and one validator.
interface IMaterialSvmClosureSink
{
    [mutating] void setShadingNormal(float3 geometricNormal, float3 shadingNormal);
    [mutating] void diffuse(SampledSpectrum weight, DiffuseLobe lobe, float3 normal);
    [mutating] void conductor(SampledSpectrum weight, ConductorLobe lobe, float3 normal);
    [mutating] void dielectric(SampledSpectrum weight, DielectricLobe lobe, float3 normal);
    void emission(float3 value);
    [mutating] bool prepare(ByteAddressBuffer energyLuts);
};

struct RaytracerMaterialSink : IMaterialSvmClosureSink
{
    NoorRayShaderData bsdf;

    [mutating] void setShadingNormal(float3 geometricNormal, float3 shadingNormal)
    { bsdf.setShadingNormal(geometricNormal, shadingNormal); }
    [mutating] void diffuse(SampledSpectrum weight, DiffuseLobe lobe, float3 normal)
    { bsdf.addDiffuse(weight, lobe, normal); }
    [mutating] void conductor(SampledSpectrum weight, ConductorLobe lobe, float3 normal)
    { bsdf.addConductor(weight, lobe, normal); }
    [mutating] void dielectric(SampledSpectrum weight, DielectricLobe lobe, float3 normal)
    { bsdf.addDielectric(weight, lobe, normal); }
    void emission(float3 value) {}
    [mutating] bool prepare(ByteAddressBuffer energyLuts)
    { return bsdf.prepare(energyLuts); }
};

// Used for sampled emissive surfaces. Avoid constructing a second composite
// BSDF in the same closest-hit invocation; only MaterialSvmResult::emission
// is consumed by direct-light estimation.
struct EmissionOnlyMaterialSink : IMaterialSvmClosureSink
{
    [mutating] void setShadingNormal(float3 geometricNormal, float3 shadingNormal) {}
    [mutating] void diffuse(SampledSpectrum weight, DiffuseLobe lobe, float3 normal) {}
    [mutating] void conductor(SampledSpectrum weight, ConductorLobe lobe, float3 normal) {}
    [mutating] void dielectric(SampledSpectrum weight, DielectricLobe lobe, float3 normal) {}
    void emission(float3 value) {}
    [mutating] bool prepare(ByteAddressBuffer energyLuts) { return true; }
};

bool evaluateMaterialSvm<T : IMaterialSvmTextures, C : IMaterialSvmClosureSink>(
    T textures, inout C sink, ByteAddressBuffer words,
    uint bytecodeOffset, uint bytecodeLength, MaterialSvmContext context,
    SampledWavelengths wavelengths, ByteAddressBuffer spectralTables,
    ByteAddressBuffer energyLuts, bool exiting,
    out MaterialSvmResult result)
{
    // MaterialX programs may read values produced by earlier graph nodes;
    // initialize every slot so an unconnected/conditionally skipped input
    // cannot feed undefined register data into the raygen integrator.
    float stack[SVM_STACK_SIZE] = {};
    result.albedo = float3(1.0); result.emission = float3(0.0);
    result.emissionStrength = 0.0; result.opacity = 1.0; result.transmission = 0.0;
    result.shadingNormal = context.interpolatedNormal; result.flags = 0u;
    float3 closureWeight = float3(1.0); uint offset = 0u;
    [loop] while (offset < bytecodeLength)
    {
        uint node = svmWord(words, bytecodeOffset + offset++);
        switch (node)
        {
        case SVM_END: return sink.prepare(energyLuts);
        case SVM_CLOSURE_WEIGHT:
            closureWeight = svmColor(words, stack, svmWord(words, bytecodeOffset + offset), svmWord(words, bytecodeOffset + offset + 1u), svmWord(words, bytecodeOffset + offset + 2u)); offset += 3u; break;
        case SVM_JUMP_IF_ZERO:
        {
            const float condition = svmValue(words, stack,
                svmWord(words, bytecodeOffset + offset));
            const uint jump = svmWord(words, bytecodeOffset + offset + 1u);
            offset += 2u;
            // Jump offsets are emitted relative to the first word after the
            // NodeJump payload. Clamp malformed streams instead of allowing
            // an out-of-bounds byte-address read to become a device fault.
            if (condition <= 0.0 && jump <= bytecodeLength - offset)
                offset += jump;
            break;
        }
        case SVM_JUMP_IF_ONE:
        {
            const float condition = svmValue(words, stack,
                svmWord(words, bytecodeOffset + offset));
            const uint jump = svmWord(words, bytecodeOffset + offset + 1u);
            offset += 2u;
            if (condition >= 1.0 && jump <= bytecodeLength - offset)
                offset += jump;
            break;
        }
        case SVM_MATH:
        {
            uint op = svmWord(words, bytecodeOffset + offset);
            float a = svmValue(words, stack, svmWord(words, bytecodeOffset + offset + 1u));
            float b = svmValue(words, stack, svmWord(words, bytecodeOffset + offset + 2u));
            float c = svmValue(words, stack, svmWord(words, bytecodeOffset + offset + 3u));
            svmStoreScalar(stack, svmWord(words, bytecodeOffset + offset + 4u) & 255u, svmMath(op, a, b, c)); offset += 5u; break;
        }
        case SVM_VECTOR_MATH:
        {
            uint op = svmWord(words, bytecodeOffset + offset);
            float3 a = svmColor(words, stack, svmWord(words, bytecodeOffset + offset + 1u), svmWord(words, bytecodeOffset + offset + 2u), svmWord(words, bytecodeOffset + offset + 3u));
            float3 b = svmColor(words, stack, svmWord(words, bytecodeOffset + offset + 4u), svmWord(words, bytecodeOffset + offset + 5u), svmWord(words, bytecodeOffset + offset + 6u));
            float3 v = op == 4u ? cross(a, b) : op == 5u ? float3(dot(a, b), 0, 0) : op == 6u ? normalize(a) : op == 7u ? float3(length(a), 0, 0) : op == 8u ? float3(distance(a, b), 0, 0) : op == 9u ? reflect(-a, normalize(b)) : a + b;
            uint packed = svmWord(words, bytecodeOffset + offset + 10u);
            uint resultOffset = packed & 255u;
            if (((packed >> 8u) & 255u) != 0u) svmStoreScalar(stack, resultOffset, v.x); else svmStoreColor(stack, resultOffset, v); offset += 11u; break;
        }
        case SVM_MIX:
        {
            float f = svmValue(words, stack, svmWord(words, bytecodeOffset + offset));
            float3 a = svmColor(words, stack, svmWord(words, bytecodeOffset + offset + 1u), svmWord(words, bytecodeOffset + offset + 2u), svmWord(words, bytecodeOffset + offset + 3u));
            float3 b = svmColor(words, stack, svmWord(words, bytecodeOffset + offset + 4u), svmWord(words, bytecodeOffset + offset + 5u), svmWord(words, bytecodeOffset + offset + 6u));
            uint resultOffset = svmWord(words, bytecodeOffset + offset + 7u) & 255u;
            float3 v = svmMix(a, b, f, svmWord(words, bytecodeOffset + offset + 8u));
            if (((svmWord(words, bytecodeOffset + offset + 7u) >> 8u) & 255u) != 0u) svmStoreColor(stack, resultOffset, v); else svmStoreScalar(stack, resultOffset, v.x); offset += 9u; break;
        }
        case SVM_CLAMP:
            svmStoreScalar(stack, svmWord(words, bytecodeOffset + offset + 3u) & 255u, clamp(svmValue(words, stack, svmWord(words, bytecodeOffset + offset)), svmValue(words, stack, svmWord(words, bytecodeOffset + offset + 1u)), svmValue(words, stack, svmWord(words, bytecodeOffset + offset + 2u)))); offset += 4u; break;
        case SVM_REMAP_RANGE:
        {
            float low = svmValue(words, stack, svmWord(words, bytecodeOffset + offset + 1u)); float high = svmValue(words, stack, svmWord(words, bytecodeOffset + offset + 2u));
            float t = abs(high - low) > 1e-20 ? (svmValue(words, stack, svmWord(words, bytecodeOffset + offset)) - low) / (high - low) : 0.0;
            if (((svmWord(words, bytecodeOffset + offset + 5u) >> 8u) & 255u) != 0u) { t = saturate(t); t *= t * (3.0 - 2.0 * t); }
            svmStoreScalar(stack, svmWord(words, bytecodeOffset + offset + 5u) & 255u, lerp(svmValue(words, stack, svmWord(words, bytecodeOffset + offset + 3u)), svmValue(words, stack, svmWord(words, bytecodeOffset + offset + 4u)), t)); offset += 6u; break;
        }
        case SVM_GAMMA:
            svmStoreScalar(stack, svmWord(words, bytecodeOffset + offset + 2u) & 255u, pow(max(svmValue(words, stack, svmWord(words, bytecodeOffset + offset)), 0.0), svmValue(words, stack, svmWord(words, bytecodeOffset + offset + 1u)))); offset += 3u; break;
        case SVM_SEPARATE_COLOR:
        {
            float3 v = svmColor(words, stack, svmWord(words, bytecodeOffset + offset), svmWord(words, bytecodeOffset + offset + 1u), svmWord(words, bytecodeOffset + offset + 2u)); if (svmWord(words, bytecodeOffset + offset + 3u) == 1u) v = svmRgbToHsv(v);
            uint packed = svmWord(words, bytecodeOffset + offset + 4u); uint x = packed & 255u, y = (packed >> 8u) & 255u, z = (packed >> 16u) & 255u; if (x != SVM_INVALID_OFFSET) svmStoreScalar(stack, x, v.x); if (y != SVM_INVALID_OFFSET) svmStoreScalar(stack, y, v.y); if (z != SVM_INVALID_OFFSET) svmStoreScalar(stack, z, v.z); offset += 5u; break;
        }
        case SVM_COMBINE_COLOR:
        {
            float3 v = float3(svmValue(words, stack, svmWord(words, bytecodeOffset + offset)), svmValue(words, stack, svmWord(words, bytecodeOffset + offset + 1u)), svmValue(words, stack, svmWord(words, bytecodeOffset + offset + 2u))); if (svmWord(words, bytecodeOffset + offset + 3u) == 1u) v = svmHsvToRgb(v); svmStoreColor(stack, svmWord(words, bytecodeOffset + offset + 4u) & 255u, v); offset += 5u; break;
        }
        case SVM_TEX_COORD:
        {
            uint source = svmWord(words, bytecodeOffset + offset), dst = svmWord(words, bytecodeOffset + offset + 1u) & 255u; float3 v = source == 0u ? float3(context.uv, 0) : source == 3u ? context.interpolatedNormal : source == 4u ? context.tangent : source == 5u ? context.bitangent : source == 6u ? context.vertexColor.xyz : source == 7u ? context.viewDirection : context.position; svmStoreColor(stack, dst, v); if (source == 8u) svmStoreScalar(stack, dst + 3u, context.vertexColor.w); offset += 2u; break;
        }
        case SVM_IMAGE_TEXTURE:
        {
            const uint base = bytecodeOffset + offset;
            const int textureSlot = asint(svmWord(words, base));
            const uint uvX = svmWord(words, base + 4u);
            const uint uvY = svmWord(words, base + 8u);
            const uint colorOffset = svmWord(words, base + 12u) & 255u;
            const uint alphaToken = svmWord(words, base + 16u);
            const uint alphaOffset = alphaToken & 255u;
            const uint addressU = svmWord(words, base + 20u) & 255u;
            const uint addressV = svmWord(words, base + 24u) & 255u;
            const uint filterType = svmWord(words, base + 28u) & 255u;
            const float2 uv = float2(svmValue(words, stack, uvX),
                svmValue(words, stack, uvY));
            const float4 value = textureSlot < 0
                ? float4(0.0) : textures.sample(uint(textureSlot),
                    uv, filterType, addressU, addressV);
            svmStoreColor(stack, colorOffset, value.xyz);
            // InvalidOffset is 255; avoid writing outside the fixed stack
            // for programs that only request RGB.
            if (alphaOffset != SVM_INVALID_OFFSET)
                svmStoreScalar(stack, alphaOffset, value.w);
            offset += 8u;
            break;
        }
        case SVM_NOISE:
        {
            const uint base = bytecodeOffset + offset;
            const float3 p = svmColor(words, stack, svmWord(words, base),
                svmWord(words, base + 1u), svmWord(words, base + 2u));
            const float3 amplitude = svmColor(words, stack, svmWord(words, base + 3u),
                svmWord(words, base + 4u), svmWord(words, base + 5u));
            const float pivot = svmValue(words, stack, svmWord(words, base + 6u));
            const uint packed = svmWord(words, base + 7u);
            const uint dst = packed & 255u;
            const bool vectorResult = ((packed >> 8u) & 255u) != 0u;
            const bool is2d = ((packed >> 16u) & 255u) != 0u;
            const float value = (is2d ? svmPerlin2d(p.xy) : svmPerlin3d(p));
            if (vectorResult)
                svmStoreColor(stack, dst, amplitude * value + pivot.xxx);
            else
                svmStoreScalar(stack, dst, amplitude.x * value + pivot);
            offset += 8u;
            break;
        }
        case SVM_FRACTAL_NOISE:
        {
            const uint base = bytecodeOffset + offset;
            const float3 p = svmColor(words, stack, svmWord(words, base),
                svmWord(words, base + 1u), svmWord(words, base + 2u));
            const float3 amplitude = svmColor(words, stack, svmWord(words, base + 3u),
                svmWord(words, base + 4u), svmWord(words, base + 5u));
            const int octaves = int(svmValue(words, stack, svmWord(words, base + 6u)));
            const float lacunarity = svmValue(words, stack, svmWord(words, base + 7u));
            const float diminish = svmValue(words, stack, svmWord(words, base + 8u));
            const uint packed = svmWord(words, base + 9u);
            const uint dst = packed & 255u;
            const bool vectorResult = ((packed >> 8u) & 255u) != 0u;
            const bool is2d = ((packed >> 16u) & 255u) != 0u;
            const float value = svmFractal(p, is2d, octaves, lacunarity, diminish);
            if (vectorResult)
                svmStoreColor(stack, dst, amplitude * value);
            else
                svmStoreScalar(stack, dst, amplitude.x * value);
            offset += 10u;
            break;
        }
        case SVM_CHECKER:
        {
            const uint base = bytecodeOffset + offset;
            const float3 color1 = svmColor(words, stack, svmWord(words, base),
                svmWord(words, base + 1u), svmWord(words, base + 2u));
            const float3 color2 = svmColor(words, stack, svmWord(words, base + 3u),
                svmWord(words, base + 4u), svmWord(words, base + 5u));
            const float u = svmValue(words, stack, svmWord(words, base + 6u))
                * svmValue(words, stack, svmWord(words, base + 8u))
                - svmValue(words, stack, svmWord(words, base + 10u));
            const float v = svmValue(words, stack, svmWord(words, base + 7u))
                * svmValue(words, stack, svmWord(words, base + 9u))
                - svmValue(words, stack, svmWord(words, base + 11u));
            const float parity = fmod(floor(u) + floor(v), 2.0);
            svmStoreColor(stack, svmWord(words, base + 12u) & 255u,
                parity == 0.0 ? color2 : color1);
            offset += 13u;
            break;
        }
        case SVM_GRADIENT:
        {
            const uint base = bytecodeOffset + offset;
            const float3 p = svmColor(words, stack, svmWord(words, base),
                svmWord(words, base + 1u), svmWord(words, base + 2u));
            const uint type = svmWord(words, base + 3u);
            float value = p.x;
            if (type == 1u) value = p.x * p.x;
            else if (type == 2u) { const float x = saturate(p.x); value = x * x * (3.0 - 2.0 * x); }
            else if (type == 3u) value = 0.5 * (p.x + p.y);
            else if (type == 4u) value = atan2(p.y, p.x) / 6.28318530718 + 0.5;
            else if (type >= 5u) { value = max(1.0 - length(p), 0.0); if (type == 6u) value *= value; }
            svmStoreScalar(stack, svmWord(words, base + 4u) & 255u, value);
            offset += 5u;
            break;
        }
        case SVM_WORLEY:
        {
            const uint base = bytecodeOffset + offset;
            const float3 p = svmColor(words, stack, svmWord(words, base),
                svmWord(words, base + 1u), svmWord(words, base + 2u));
            const uint packed = svmWord(words, base + 5u);
            const uint dst = packed & 255u;
            const bool vectorResult = ((packed >> 8u) & 255u) != 0u;
            const bool is2d = ((packed >> 16u) & 255u) != 0u;
            const float value = svmWorleyNoise(p, is2d);
            if (vectorResult) svmStoreColor(stack, dst, value.xxx);
            else svmStoreScalar(stack, dst, value);
            offset += 6u;
            break;
        }
        case SVM_CELL:
        {
            const uint base = bytecodeOffset + offset;
            const float3 p = svmColor(words, stack, svmWord(words, base),
                svmWord(words, base + 1u), svmWord(words, base + 2u));
            const uint packed = svmWord(words, base + 3u);
            const bool is2d = ((packed >> 8u) & 255u) != 0u;
            svmStoreScalar(stack, packed & 255u, svmCellNoise(p, is2d));
            offset += 4u;
            break;
        }
        case SVM_UNIFIED_NOISE:
        {
            const uint base = bytecodeOffset + offset;
            const float3 p0 = svmColor(words, stack, svmWord(words, base),
                svmWord(words, base + 1u), svmWord(words, base + 2u));
            const float3 freq = svmColor(words, stack, svmWord(words, base + 3u),
                svmWord(words, base + 4u), svmWord(words, base + 5u));
            const float3 shift = svmColor(words, stack, svmWord(words, base + 6u),
                svmWord(words, base + 7u), svmWord(words, base + 8u));
            // The frequency fields occupy words 3..5 and the offset fields
            // words 6..8 in NodeUnifiedNoiseTexture.
            const float3 position = p0 * freq + shift;
            const float jitter = svmValue(words, stack, svmWord(words, base + 9u));
            const float outMin = svmValue(words, stack, svmWord(words, base + 10u));
            const float outMax = svmValue(words, stack, svmWord(words, base + 11u));
            const float clampOutput = svmValue(words, stack, svmWord(words, base + 12u));
            const int octaves = int(svmValue(words, stack, svmWord(words, base + 13u)));
            const float lacunarity = svmValue(words, stack, svmWord(words, base + 14u));
            const float diminish = svmValue(words, stack, svmWord(words, base + 15u));
            const int noiseType = int(svmValue(words, stack, svmWord(words, base + 16u)));
            const uint packed = svmWord(words, base + 18u);
            const bool is2d = ((packed >> 8u) & 255u) != 0u;
            float value = noiseType == 1 ? svmCellNoise(position, is2d)
                : noiseType == 2 ? svmWorleyNoise(position, is2d)
                : noiseType == 3 ? svmFractal(position, is2d, octaves, lacunarity, diminish)
                : 0.5 * (is2d ? svmPerlin2d(position.xy) : svmPerlin3d(position)) + 0.5;
            value = outMin + value * (outMax - outMin);
            if (clampOutput != 0.0) value = clamp(value, min(outMin, outMax), max(outMin, outMax));
            svmStoreScalar(stack, packed & 255u, value);
            offset += 19u;
            break;
        }
        case SVM_NORMAL_MAP:
        {
            const uint base = bytecodeOffset + offset;
            float3 value = svmColor(words, stack, svmWord(words, base),
                svmWord(words, base + 1u), svmWord(words, base + 2u));
            value = dot(value, value) == 0.0 ? float3(0.0, 0.0, 1.0) : value * 2.0 - 1.0;
            const float2 scale = float2(svmValue(words, stack, svmWord(words, base + 3u)),
                svmValue(words, stack, svmWord(words, base + 4u)));
            const float3 normal = svmColor(words, stack, svmWord(words, base + 5u),
                svmWord(words, base + 6u), svmWord(words, base + 7u));
            const float3 tangent = svmColor(words, stack, svmWord(words, base + 8u),
                svmWord(words, base + 9u), svmWord(words, base + 10u));
            const float3 bitangent = svmColor(words, stack, svmWord(words, base + 11u),
                svmWord(words, base + 12u), svmWord(words, base + 13u));
            const float3 mapped = normalize(tangent * value.x * scale.x
                + bitangent * value.y * scale.y + normal * value.z);
            svmStoreColor(stack, svmWord(words, base + 14u) & 255u, mapped);
            offset += 15u;
            break;
        }
        case SVM_CLOSURE_DIFFUSE:
        {
            const uint base = bytecodeOffset + offset;
            const float3 color = svmColor(words, stack, svmWord(words, base),
                svmWord(words, base + 1u), svmWord(words, base + 2u));
            const float roughness = svmValue(words, stack, svmWord(words, base + 3u));
            const uint normalX = svmWord(words, base + 4u);
            const float3 normal = svmInvalidInput(normalX)
                ? context.interpolatedNormal
                : svmColor(words, stack, normalX, svmWord(words, base + 5u), svmWord(words, base + 6u));
            result.albedo = closureWeight * color;
            result.shadingNormal = normalize(normal);
            result.flags |= SVM_HAS_SHADING_NORMAL;
            DiffuseLobe lobe = makeDiffuseLobe();
            lobe.albedo = rgbAlbedoToSpectrum(spectralTables, color, wavelengths);
            lobe.roughness = saturate(roughness);
            lobe.burley = svmWord(words, base + 7u) != 0u;
            lobe.translucent = svmWord(words, base + 8u) != 0u;
            sink.diffuse(rgbAlbedoToSpectrum(spectralTables, closureWeight,
                wavelengths), lobe, svmInvalidInput(normalX) ? float3(0.0) : normal);
            offset += 9u;
            break;
        }
        case SVM_CLOSURE_CONDUCTOR:
        {
            const uint base = bytecodeOffset + offset;
            const float3 color = svmColor(words, stack, svmWord(words, base),
                svmWord(words, base + 1u), svmWord(words, base + 2u));
            const float3 normal = svmColor(words, stack, svmWord(words, base + 11u),
                svmWord(words, base + 12u), svmWord(words, base + 13u));
            result.albedo = closureWeight * color;
            result.shadingNormal = normalize(normal);
            result.flags |= SVM_HAS_SHADING_NORMAL;
            ConductorLobe lobe = makeConductorLobe();
            lobe.eta = rgbAlbedoToSpectrum(spectralTables,
                svmColor(words, stack, svmWord(words, base + 5u),
                    svmWord(words, base + 6u), svmWord(words, base + 7u)), wavelengths);
            lobe.extinction = rgbAlbedoToSpectrum(spectralTables,
                svmColor(words, stack, svmWord(words, base + 8u),
                    svmWord(words, base + 9u), svmWord(words, base + 10u)), wavelengths);
            lobe.roughness = saturate(svmValue(words, stack,
                svmWord(words, base + 3u)));
            const uint normalX = svmWord(words, base + 11u);
            sink.conductor(rgbAlbedoToSpectrum(spectralTables,
                    closureWeight * color, wavelengths), lobe,
                svmInvalidInput(normalX) ? float3(0.0) : normal);
            offset += 14u;
            break;
        }
        case SVM_CLOSURE_DIELECTRIC:
        {
            const uint base = bytecodeOffset + offset;
            const float3 color = svmColor(words, stack, svmWord(words, base),
                svmWord(words, base + 1u), svmWord(words, base + 2u));
            const float transmission = svmValue(words, stack, svmWord(words, base + 12u));
            result.albedo = closureWeight * color;
            result.transmission = saturate(transmission > 0.5 ? max(color.x, max(color.y, color.z)) : 0.0);
            const uint normalBase = base + 13u;
            const uint normalX = svmWord(words, normalBase);
            if (!svmInvalidInput(normalX))
            {
                result.shadingNormal = normalize(svmColor(words, stack, normalX,
                    svmWord(words, normalBase + 1u), svmWord(words, normalBase + 2u)));
                result.flags |= SVM_HAS_SHADING_NORMAL;
            }
            const SampledSpectrum tint = rgbAlbedoToSpectrum(spectralTables,
                color, wavelengths);
            DielectricLobe lobe = makeDielectricLobe();
            const bool hasReflection = transmission < 0.5 || transmission > 1.5;
            const bool hasTransmission = transmission > 0.5;
            lobe.reflectionTint = hasReflection ? tint : SampledSpectrum(0.0);
            lobe.transmissionTint = hasTransmission ? tint : SampledSpectrum(0.0);
            lobe.roughness = saturate(svmValue(words, stack,
                svmWord(words, base + 3u)));
            lobe.ior = max(svmValue(words, stack, svmWord(words, base + 4u)), 1.0);
            const bool spectralIor = svmWord(words, base + 5u) != 0u
                && hasTransmission;
            if (spectralIor)
            {
                const float3 b = svmColor(words, stack,
                    svmWord(words, base + 6u), svmWord(words, base + 7u),
                    svmWord(words, base + 8u));
                const float3 c = svmColor(words, stack,
                    svmWord(words, base + 9u), svmWord(words, base + 10u),
                    svmWord(words, base + 11u));
                lobe.spectralIor = sellmeierSpectrum(b, c, wavelengths);
                lobe.useSpectralIor = true;
            }
            lobe.exiting = exiting;
            sink.dielectric(rgbAlbedoToSpectrum(spectralTables,
                    closureWeight, wavelengths), lobe,
                svmInvalidInput(normalX) ? float3(0.0) : result.shadingNormal);
            offset += 16u;
            break;
        }
        case SVM_CLOSURE_SHEEN:
        {
            const uint base = bytecodeOffset + offset;
            const float3 color = svmColor(words, stack, svmWord(words, base),
                svmWord(words, base + 1u), svmWord(words, base + 2u));
            result.albedo = closureWeight * color;
            DielectricLobe lobe = makeDielectricLobe();
            lobe.reflectionTint = rgbAlbedoToSpectrum(spectralTables,
                color, wavelengths);
            lobe.roughness = saturate(svmValue(words, stack,
                svmWord(words, base + 3u)));
            lobe.exiting = exiting;
            const uint normalX = svmWord(words, base + 4u);
            const float3 normal = svmInvalidInput(normalX) ? float3(0.0)
                : svmColor(words, stack, normalX, svmWord(words, base + 5u),
                    svmWord(words, base + 6u));
            sink.dielectric(rgbAlbedoToSpectrum(spectralTables,
                closureWeight, wavelengths), lobe, normal);
            offset += 7u;
            break;
        }
        case SVM_CLOSURE_SUBSURFACE:
        {
            const uint base = bytecodeOffset + offset;
            const float3 color = svmColor(words, stack, svmWord(words, base),
                svmWord(words, base + 1u), svmWord(words, base + 2u));
            result.albedo = closureWeight * color;
            DiffuseLobe lobe = makeDiffuseLobe();
            lobe.albedo = rgbAlbedoToSpectrum(spectralTables, color, wavelengths);
            lobe.roughness = saturate(svmValue(words, stack,
                svmWord(words, base + 3u)));
            const uint normalX = svmWord(words, base + 4u);
            const float3 normal = svmInvalidInput(normalX) ? float3(0.0)
                : svmColor(words, stack, normalX, svmWord(words, base + 5u),
                    svmWord(words, base + 6u));
            sink.diffuse(rgbAlbedoToSpectrum(spectralTables,
                closureWeight, wavelengths), lobe, normal);
            offset += 7u;
            break;
        }
        case SVM_CLOSURE_EDF:
        {
            float3 value = closureWeight * svmColor(words, stack, svmWord(words, bytecodeOffset + offset), svmWord(words, bytecodeOffset + offset + 1u), svmWord(words, bytecodeOffset + offset + 2u)) * max(svmValue(words, stack, svmWord(words, bytecodeOffset + offset + 3u)), 0.0); result.emission += value; result.emissionStrength = 1.0; result.flags |= value.x != 0 || value.y != 0 || value.z != 0 ? SVM_HAS_EMISSION : 0u; sink.emission(value); offset += 4u; break;
        }
        case SVM_CLOSURE_OPEN_PBR:
        {
            const uint base = bytecodeOffset + offset;
            const float3 baseColor = svmColor(words, stack, svmWord(words, base),
                svmWord(words, base + 1u), svmWord(words, base + 2u));
            const float baseWeight = saturate(svmValue(words, stack, svmWord(words, base + 3u)));
            const float diffuseRoughness = saturate(svmValue(words, stack,
                svmWord(words, base + 4u)));
            const float metalness = saturate(svmValue(words, stack, svmWord(words, base + 5u)));
            const float specularWeight = saturate(svmValue(words, stack,
                svmWord(words, base + 6u)));
            const float roughness = saturate(svmValue(words, stack,
                svmWord(words, base + 7u)));
            const float ior = max(svmValue(words, stack,
                svmWord(words, base + 8u)), 1.0);
            const bool disneyPrincipled = svmWord(words, base + 53u) != 0u;
            const float3 specularColor = svmColor(words, stack,
                svmWord(words, base + 16u), svmWord(words, base + 17u),
                svmWord(words, base + 18u));
            const float transmissionWeight = saturate(svmValue(words, stack, svmWord(words, base + 19u)));
            const float3 transmissionColor = svmColor(words, stack, svmWord(words, base + 20u),
                svmWord(words, base + 21u), svmWord(words, base + 22u));
            const float subsurfaceWeight = saturate(svmValue(words, stack, svmWord(words, base + 23u)));
            const float3 subsurfaceColor = svmColor(words, stack, svmWord(words, base + 24u),
                svmWord(words, base + 25u), svmWord(words, base + 26u));
            const float fuzzWeight = saturate(svmValue(words, stack,
                svmWord(words, base + 27u)));
            const float3 fuzzColor = svmColor(words, stack,
                svmWord(words, base + 28u), svmWord(words, base + 29u),
                svmWord(words, base + 30u));
            const float fuzzRoughness = saturate(svmValue(words, stack,
                svmWord(words, base + 31u)));
            const float coatWeight = saturate(svmValue(words, stack,
                svmWord(words, base + 32u)));
            const float3 coatColor = svmColor(words, stack,
                svmWord(words, base + 33u), svmWord(words, base + 34u),
                svmWord(words, base + 35u));
            const float coatRoughness = saturate(svmValue(words, stack,
                svmWord(words, base + 36u)));
            const float coatIor = max(svmValue(words, stack,
                svmWord(words, base + 37u)), 1.0);
            const float3 diffuse = baseColor * baseWeight * (1.0 - metalness)
                * (1.0 - transmissionWeight);
            result.albedo = diffuse * (1.0 - subsurfaceWeight)
                + subsurfaceColor * baseWeight * subsurfaceWeight;
            result.transmission = transmissionWeight * max(transmissionColor.x,
                max(transmissionColor.y, transmissionColor.z));
            const float3 emissionColor = svmColor(words, stack, svmWord(words, base + 48u),
                svmWord(words, base + 49u), svmWord(words, base + 50u));
            const float emissionStrength = max(svmValue(words, stack, svmWord(words, base + 51u)), 0.0);
            result.emission += closureWeight * emissionColor * emissionStrength;
            if (emissionStrength > 0.0) { result.emissionStrength = 1.0; result.flags |= SVM_HAS_EMISSION; }
            result.opacity = saturate(svmValue(words, stack, svmWord(words, base + 52u)));
            if (result.opacity < 1.0) result.flags |= SVM_HAS_CUTOUT;
            const uint normalX = svmWord(words, base + 45u);
            if (!svmInvalidInput(normalX))
            {
                result.shadingNormal = normalize(svmColor(words, stack, normalX,
                    svmWord(words, base + 46u), svmWord(words, base + 47u)));
                result.flags |= SVM_HAS_SHADING_NORMAL;
                sink.setShadingNormal(context.geometricNormal,
                    result.shadingNormal);
            }

            const SampledSpectrum baseSpectrum = rgbAlbedoToSpectrum(
                spectralTables, baseColor, wavelengths);
            const SampledSpectrum specularSpectrum = rgbAlbedoToSpectrum(
                spectralTables, specularColor, wavelengths);
            const SampledSpectrum transmissionSpectrum = rgbAlbedoToSpectrum(
                spectralTables, transmissionColor, wavelengths);
            const SampledSpectrum subsurfaceSpectrum = rgbAlbedoToSpectrum(
                spectralTables, subsurfaceColor, wavelengths);
            const SampledSpectrum fuzzSpectrum = rgbAlbedoToSpectrum(
                spectralTables, fuzzColor, wavelengths);
            const SampledSpectrum coatSpectrum = rgbAlbedoToSpectrum(
                spectralTables, coatColor, wavelengths);
            const SampledSpectrum weight = rgbAlbedoToSpectrum(
                spectralTables, closureWeight, wavelengths);
            const bool spectralTransmission = svmWord(words, base + 9u) != 0u
                && transmissionWeight > 0.0;
            SampledSpectrum specularIor = SampledSpectrum(ior);
            if (spectralTransmission)
            {
                const float3 b = svmColor(words, stack,
                    svmWord(words, base + 10u), svmWord(words, base + 11u),
                    svmWord(words, base + 12u));
                const float3 c = svmColor(words, stack,
                    svmWord(words, base + 13u), svmWord(words, base + 14u),
                    svmWord(words, base + 15u));
                specularIor = sellmeierSpectrum(b, c, wavelengths);
            }

            if (baseWeight > 0.0 && metalness < 1.0
                && transmissionWeight < 1.0)
            {
                const float baseDiffuseWeight = baseWeight * (1.0 - metalness)
                    * (1.0 - transmissionWeight);
                DiffuseLobe diffuseLobe = makeDiffuseLobe();
                diffuseLobe.albedo = baseSpectrum;
                diffuseLobe.roughness = diffuseRoughness;
                diffuseLobe.burley = true;
                sink.diffuse(weight * (baseDiffuseWeight
                    * (1.0 - subsurfaceWeight)), diffuseLobe, float3(0.0));
                if (subsurfaceWeight > 0.0)
                {
                    DiffuseLobe subsurfaceLobe = makeDiffuseLobe();
                    subsurfaceLobe.albedo = subsurfaceSpectrum;
                    subsurfaceLobe.roughness = diffuseRoughness;
                    subsurfaceLobe.burley = true;
                    sink.diffuse(weight * (baseDiffuseWeight * subsurfaceWeight),
                        subsurfaceLobe, float3(0.0));
                }
            }
            if (metalness > 0.0)
            {
                ConductorLobe conductorLobe = makeConductorLobe();
                [unroll]
                for (int i = 0; i < NrSpectrumSamples; ++i)
                {
                    conductorLobe.eta[i] = iorFromNormalReflectance(
                        clamp(baseSpectrum[i], 0.0, 0.9999));
                    conductorLobe.extinction[i] = 0.0;
                }
                conductorLobe.roughness = roughness;
                sink.conductor(weight * (baseWeight * metalness),
                    conductorLobe, float3(0.0));
            }
            if (metalness < 1.0)
            {
                if (disneyPrincipled && transmissionWeight > 0.0)
                {
                    DielectricLobe glass = makeDielectricLobe();
                    glass.reflectionTint = specularWeight > 0.0
                        ? SampledSpectrum(1.0) : SampledSpectrum(0.0);
                    glass.transmissionTint = transmissionSpectrum
                        * transmissionWeight;
                    glass.roughness = roughness;
                    glass.ior = specularWeight > 0.0
                        ? iorFromNormalReflectance(0.08 * specularWeight) : ior;
                    glass.spectralIor = specularIor;
                    glass.useSpectralIor = spectralTransmission;
                    glass.exiting = exiting;
                    sink.dielectric(weight * (baseWeight * (1.0 - metalness)),
                        glass, float3(0.0));
                }
                else
                {
                    if (specularWeight > 0.0)
                    {
                        DielectricLobe reflection = makeDielectricLobe();
                        reflection.reflectionTint = disneyPrincipled
                            ? SampledSpectrum(1.0) : specularSpectrum;
                        reflection.transmissionTint = SampledSpectrum(0.0);
                        reflection.roughness = roughness;
                        reflection.ior = disneyPrincipled
                            ? iorFromNormalReflectance(0.08 * specularWeight) : ior;
                        reflection.spectralIor = specularIor;
                        reflection.useSpectralIor = spectralTransmission;
                        reflection.exiting = exiting;
                        sink.dielectric(weight * (baseWeight * (1.0 - metalness)
                            * (disneyPrincipled ? 1.0 : specularWeight)),
                            reflection, float3(0.0));
                    }
                    if (transmissionWeight > 0.0)
                    {
                        DielectricLobe transmissionLobe = makeDielectricLobe();
                        transmissionLobe.reflectionTint = SampledSpectrum(0.0);
                        transmissionLobe.transmissionTint = transmissionSpectrum;
                        transmissionLobe.roughness = roughness;
                        transmissionLobe.ior = ior;
                        transmissionLobe.spectralIor = specularIor;
                        transmissionLobe.useSpectralIor = spectralTransmission;
                        transmissionLobe.exiting = exiting;
                        sink.dielectric(weight * (baseWeight * (1.0 - metalness)
                            * transmissionWeight), transmissionLobe,
                            float3(0.0));
                    }
                }
            }
            if (coatWeight > 0.0)
            {
                DielectricLobe coat = makeDielectricLobe();
                coat.reflectionTint = coatSpectrum;
                coat.roughness = coatRoughness;
                coat.ior = coatIor;
                if (svmWord(words, base + 38u) != 0u)
                {
                    const float3 b = svmColor(words, stack,
                        svmWord(words, base + 39u), svmWord(words, base + 40u),
                        svmWord(words, base + 41u));
                    const float3 c = svmColor(words, stack,
                        svmWord(words, base + 42u), svmWord(words, base + 43u),
                        svmWord(words, base + 44u));
                    coat.spectralIor = sellmeierSpectrum(b, c, wavelengths);
                    coat.useSpectralIor = true;
                }
                coat.exiting = exiting;
                sink.dielectric(weight * coatWeight, coat, float3(0.0));
            }
            if (fuzzWeight > 0.0)
            {
                DielectricLobe fuzz = makeDielectricLobe();
                fuzz.reflectionTint = fuzzSpectrum;
                fuzz.roughness = fuzzRoughness;
                fuzz.ior = 1.2;
                fuzz.exiting = exiting;
                sink.dielectric(weight * fuzzWeight, fuzz, float3(0.0));
            }
            offset += 54u;
            break;
        }
        case SVM_SURFACE_OUTPUT:
            result.opacity = saturate(svmValue(words, stack, svmWord(words, bytecodeOffset + offset))); if (result.opacity < 1.0) result.flags |= SVM_HAS_CUTOUT; offset += 1u; break;
        default: return false;
        }
    }
    return false;
}
