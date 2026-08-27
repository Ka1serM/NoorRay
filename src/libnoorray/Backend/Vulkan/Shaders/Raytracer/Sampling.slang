#pragma once

// Sampling for the Vulkan path-tracing path.
//
// A direct port of Rendering/Sampling/{OwenSobolSampler,Sampler,RandomSampler}.h
// and Rendering/MisHeuristic.h. The sequence must match the CPU implementation
// exactly, so the constants and bit operations are carried over verbatim.

// Joe-Kuo direction numbers for Sobol dimensions 2 through 6.
static const uint OwenSobolDirectionNumbers[5][32] = {
    0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
    0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000, 0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
    0x80008000, 0xc000c000, 0xa000a000, 0xf000f000, 0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
    0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0, 0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,
    0x80000000, 0xc0000000, 0x60000000, 0x90000000, 0xe8000000, 0x5c000000, 0x8e000000, 0xc5000000,
    0x68800000, 0x9cc00000, 0xee600000, 0x55900000, 0x80680000, 0xc09c0000, 0x60ee0000, 0x90550000,
    0xe8808000, 0x5cc0c000, 0x8e606000, 0xc5909000, 0x6868e800, 0x9c9c5c00, 0xeeee8e00, 0x5555c500,
    0x8000e880, 0xc0005cc0, 0x60008e60, 0x9000c590, 0xe8006868, 0x5c009c9c, 0x8e00eeee, 0xc5005555,
    0x80000000, 0xc0000000, 0x20000000, 0x50000000, 0xf8000000, 0x74000000, 0xa2000000, 0x93000000,
    0xd8800000, 0x25400000, 0x59e00000, 0xe6d00000, 0x78080000, 0xb40c0000, 0x82020000, 0xc3050000,
    0x208f8000, 0x51474000, 0xfbea2000, 0x75d93000, 0xa0858800, 0x914e5400, 0xdbe79e00, 0x25db6d00,
    0x58800080, 0xe54000c0, 0x79e00020, 0xb6d00050, 0x800800f8, 0xc00c0074, 0x200200a2, 0x50050093,
    0x80000000, 0x40000000, 0x20000000, 0xb0000000, 0xf8000000, 0xdc000000, 0x7a000000, 0x9d000000,
    0x5a800000, 0x2fc00000, 0xa1600000, 0xf0b00000, 0xda880000, 0x6fc40000, 0x81620000, 0x40bb0000,
    0x22878000, 0xb3c9c000, 0xfb65a000, 0xddb2d000, 0x78022800, 0x9c0b3c00, 0x5a0fb600, 0x2d0ddb00,
    0xa2878080, 0xf3c9c040, 0xdb65a020, 0x6db2d0b0, 0x800228f8, 0x400b3cdc, 0x200fb67a, 0xb00ddb9d,
    0x80000000, 0x40000000, 0x60000000, 0x30000000, 0xc8000000, 0x24000000, 0x56000000, 0xfb000000,
    0xe0800000, 0x70400000, 0xa8600000, 0x14300000, 0x9ec80000, 0xdf240000, 0xb6d60000, 0x8bbb0000,
    0x48008000, 0x64004000, 0x36006000, 0xcb003000, 0x2880c800, 0x54402400, 0xfe605600, 0xef30fb00,
    0x7e48e080, 0xaf647040, 0x1eb6a860, 0x9f8b1430, 0xd6c81ec8, 0xbb249f24, 0x80d6d6d6, 0x40bbbbbb,
};

// ---------------------------------------------------------------------------
// Hashing (RandomSampler.h)
// ---------------------------------------------------------------------------

float unitFloatFromBits(uint bits)
{
    const uint DiscardedLowBits = 8u;
    const float BinCenter = 0.5;
    const float InverseBinCount = 1.0 / 16777216.0;
    return (float(bits >> DiscardedLowBits) + BinCenter) * InverseBinCount;
}

uint pcgHash(uint value)
{
    // PCG RXS-M-XS constants from O'Neill's PCG reference implementation.
    const uint Multiplier = 747796405u;
    const uint Increment = 2891336453u;
    const uint OutputMultiplier = 277803737u;
    const uint VariableShiftBase = 4u;
    const uint VariableShiftSource = 28u;
    value = value * Multiplier + Increment;
    const uint shift = (value >> VariableShiftSource) + VariableShiftBase;
    return ((value >> shift) ^ value) * OutputMultiplier ^ value;
}

uint hashCombine32(uint first, uint second)
{
    return pcgHash(first ^ pcgHash(second));
}

uint sampleDimensionSeed(uint scramble, uint dimension)
{
    return hashCombine32(scramble, dimension);
}

float hashFloat(uint value)
{
    return unitFloatFromBits(pcgHash(value));
}

// ---------------------------------------------------------------------------
// Owen-scrambled Sobol (OwenSobolSampler.h)
// ---------------------------------------------------------------------------

uint sobolReverseBits(uint value)
{
    const uint AlternatingBits = 0x55555555u;
    const uint AlternatingPairs = 0x33333333u;
    const uint AlternatingNibbles = 0x0f0f0f0fu;
    const uint AlternatingBytes = 0x00ff00ffu;
    value = ((value & AlternatingBits) << 1u) | ((value >> 1u) & AlternatingBits);
    value = ((value & AlternatingPairs) << 2u) | ((value >> 2u) & AlternatingPairs);
    value = ((value & AlternatingNibbles) << 4u) | ((value >> 4u) & AlternatingNibbles);
    value = ((value & AlternatingBytes) << 8u) | ((value >> 8u) & AlternatingBytes);
    return (value << 16u) | (value >> 16u);
}

uint sobolSampleBits(uint sampleIndex, uint dimension)
{
    if (dimension == 0u)
        return sobolReverseBits(sampleIndex);

    uint value = 0u;
    uint index = sampleIndex;
    while (index != 0u)
    {
        const uint bit = firstbitlow(index);
        value ^= OwenSobolDirectionNumbers[dimension - 1u][bit];
        index &= index - 1u;
    }
    return value;
}

uint fastOwenScramble(uint value, uint seed)
{
    // Fixed permutation constants from Burley's FastOwenScrambler.
    const uint InitialMultiplier = 0x3d20adeau;
    const uint FirstXorMultiplier = 0x05526c56u;
    const uint SecondXorMultiplier = 0x53a22864u;
    value = sobolReverseBits(value);
    value ^= value * InitialMultiplier;
    value += seed;
    value *= (seed >> 16u) | 1u;
    value ^= value * FirstXorMultiplier;
    value ^= value * SecondXorMultiplier;
    return sobolReverseBits(value);
}

float owenSobolScramble(uint bits, uint seed)
{
    return unitFloatFromBits(fastOwenScramble(bits, seed));
}

// Cycles-style path sampling: small, independently scrambled Sobol blocks
// rather than an ever-growing global dimension index.
struct PathSampler
{
    uint sampleIndex;
    uint scramble;

    float next1D(uint dimension)
    {
        const uint seed = sampleDimensionSeed(scramble, dimension);
        return owenSobolScramble(sobolSampleBits(sampleIndex, dimension % 6u), seed);
    }

    float2 next2D(uint dimension)
    {
        return float2(next1D(dimension), next1D(dimension + 1u));
    }
};

// Sequential PCG stream, matching randomFloat(uint32_t&) in RandomSampler.h.
// The closures consume it exactly as the renderer expects.
struct Rng
{
    uint state;

    [mutating] float nextFloat()
    {
        state = pcgHash(state);
        return unitFloatFromBits(state);
    }

    [mutating] float2 nextFloat2()
    {
        const float x = nextFloat();
        const float y = nextFloat();
        return float2(x, y);
    }
};

// ---------------------------------------------------------------------------
// MIS (MisHeuristic.h)
// ---------------------------------------------------------------------------

float balanceHeuristic(float a, float b)
{
    return a / max(a + b, 1e-20);
}

float powerHeuristic(float a, float b)
{
    const float a2 = a * a;
    const float b2 = b * b;
    return a2 / max(a2 + b2, 1e-20);
}
