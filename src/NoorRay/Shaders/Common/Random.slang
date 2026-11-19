import NoorRHI;

uint hashPcg(uint value) {
    const uint state = value * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint hashCombine(uint seed, uint value) { return hashPcg(seed ^ (value + 0x9e3779b9u + (seed << 6) + (seed >> 2))); }

float uintToFloat(uint value) { return float(value >> 8) * 5.9604645e-8; }

//! A small generator for decisions that need no stratification.
struct Random {
    uint state;

    [mutating]
    float next() {
        state = state * 747796405u + 2891336453u;
        return uintToFloat(hashPcg(state));
    }
};

Random makeRandom(uint seed) {
    Random random;
    random.state = hashPcg(seed);
    return random;
}

//! The direction numbers of the first four Sobol dimensions after the identity.
static const uint sobolMatrices[4][32] = {
    { 0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u, 0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u, 0x00800000u, 0x00400000u, 0x00200000u,
      0x00100000u, 0x00080000u, 0x00040000u, 0x00020000u, 0x00010000u, 0x00008000u, 0x00004000u, 0x00002000u, 0x00001000u, 0x00000800u, 0x00000400u,
      0x00000200u, 0x00000100u, 0x00000080u, 0x00000040u, 0x00000020u, 0x00000010u, 0x00000008u, 0x00000004u, 0x00000002u, 0x00000001u },
    { 0x80000000u, 0xc0000000u, 0xa0000000u, 0xf0000000u, 0x88000000u, 0xcc000000u, 0xaa000000u, 0xff000000u, 0x80800000u, 0xc0c00000u, 0xa0a00000u,
      0xf0f00000u, 0x88880000u, 0xcccc0000u, 0xaaaa0000u, 0xffff0000u, 0x80008000u, 0xc000c000u, 0xa000a000u, 0xf000f000u, 0x88008800u, 0xcc00cc00u,
      0xaa00aa00u, 0xff00ff00u, 0x80808080u, 0xc0c0c0c0u, 0xa0a0a0a0u, 0xf0f0f0f0u, 0x88888888u, 0xccccccccu, 0xaaaaaaaau, 0xffffffffu },
    { 0x80000000u, 0xc0000000u, 0x60000000u, 0x90000000u, 0xe8000000u, 0x5c000000u, 0x8e000000u, 0xc5000000u, 0x68800000u, 0x9cc00000u, 0xee600000u,
      0x55900000u, 0x80680000u, 0xc09c0000u, 0x60ee0000u, 0x90550000u, 0xe8808000u, 0x5cc0c000u, 0x8e606000u, 0xc5909000u, 0x6868e800u, 0x9c9c5c00u,
      0xeeee8e00u, 0x5555c500u, 0x8000e880u, 0xc0005cc0u, 0x60008e60u, 0x9000c590u, 0xe8006868u, 0x5c009c9cu, 0x8e00eeeeu, 0xc5005555u },
    { 0x80000000u, 0xc0000000u, 0x20000000u, 0x50000000u, 0xf8000000u, 0x74000000u, 0xa2000000u, 0x93000000u, 0xd8800000u, 0x25400000u, 0x59e00000u,
      0xe6d00000u, 0x78080000u, 0xb40c0000u, 0x82020000u, 0xc3050000u, 0x208f8000u, 0x51474000u, 0xfbea2000u, 0x75d93000u, 0xa0858800u, 0x914e5400u,
      0xdbe79e00u, 0x25db6d00u, 0x58800080u, 0xe54000c0u, 0x79e00020u, 0xb6d00050u, 0x800800f8u, 0xc00c0074u, 0x200200a2u, 0x50050093u }
};

uint sobol(uint index, uint dimension) {
    uint result = 0;
    [unroll]
    for (uint bit = 0; bit < 32; ++bit)
        if (((index >> bit) & 1u) != 0)
            result ^= sobolMatrices[dimension][bit];
    return result;
}

//! Laine-Karras style nested uniform scramble (Burley 2020).
uint nestedUniformScramble(uint x, uint seed) {
    x = reversebits(x);
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return reversebits(x);
}

//! Owen-scrambled Sobol (Burley 2020, "Practical Hash-based Owen Scrambling"): four dimensions per pattern; each sample dimension draws a new shuffled, scrambled pattern.
struct SobolSampler {
    uint index;
    uint seed;
    uint dimension;

    [mutating]
    float4 next4() {
        const uint patternSeed = hashCombine(seed, dimension);
        dimension += 1;
        const uint shuffled = nestedUniformScramble(index, patternSeed);
        float4 values;
        [unroll]
        for (uint d = 0; d < 4; ++d)
            values[d] = uintToFloat(nestedUniformScramble(sobol(shuffled, d), hashCombine(patternSeed, d + 1)));
        return values;
    }

    [mutating]
    float2 next2() { return next4().xy; }

    [mutating]
    float next() { return next4().x; }
};

SobolSampler makeSobol(uint2 pixel, uint sampleIndex, uint frameSeed) {
    SobolSampler sampler;
    sampler.index = sampleIndex;
    sampler.seed = hashCombine(hashCombine(pixel.x, pixel.y), frameSeed);
    sampler.dimension = 0;
    return sampler;
}
