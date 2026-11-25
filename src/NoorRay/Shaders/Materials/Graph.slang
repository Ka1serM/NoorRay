import NoorRHI;
import SceneRecords;
import Common.Math;
__exported import Materials.GraphTrace;
__exported import Materials.GraphRadialTiling;

//! The graph host: Materials::GraphSpirv writes the body of evaluateMaterialGraph; the helpers it calls take and give float vectors and material data.

//! The graph outputs: BsdfInputs, then transparency; the splice merges types by structure, and the float member keeps this apart from ShadingContext.
struct MaterialGraphOutput {
    float4 baseColorAlpha;
    float4 metallicRoughnessIorSpecular;
    float4 specularTintAnisotropic;
    float4 normalRotation;
    float4 tangentTransmission;
    float4 coat;
    float4 coatTint;
    float4 coatNormal;
    float4 sheen;
    float4 sheenTint;
    float4 emission;
    float4 thinFilmDispersion;
    float4 closure;
    float4 subsurface;
    float transparency;
};

//! The material data of a graph: x and y the address of the parameter values, z and w the address of the texture records.
uint4 graphMaterial(Ptr<float4> parameters, Ptr<MaterialTextureRecord> textures) { return uint4(parameters.low, parameters.high, textures.low, textures.high); }

[noinline]
float4 graphParameter(uint4 material, float index) {
    Ptr<float4> parameters;
    parameters.low = material.x;
    parameters.high = material.y;
    return parameters[uint(index)];
}

//! Images store v from the top, so v flips; uvGradients are the gradients of UV map 0.
[noinline]
float4 graphTexture(uint4 material, float4 uvGradients, float slot, float2 uv) {
    Ptr<MaterialTextureRecord> textures;
    textures.low = material.z;
    textures.high = material.w;
    const MaterialTextureRecord record = textures[uint(slot)];
    const float2 transformed = float2(dot(record.row0.xy, uv) + record.row0.z, dot(record.row1.xy, uv) + record.row1.z);
    TextureHandle texture;
    texture.value = record.texture;
    SamplerHandle sampler;
    sampler.value = record.sampler;
    const float2 dx = float2(dot(record.row0.xy, uvGradients.xy), -dot(record.row1.xy, uvGradients.xy));
    const float2 dy = float2(dot(record.row0.xy, uvGradients.zw), -dot(record.row1.xy, uvGradients.zw));
    return texture.SampleGrad(sampler, float2(transformed.x, 1.0 - transformed.y), dx, dy);
}

//! Bob Jenkins' lookup3 hashes, as Blender's procedural textures use them.
uint rotateBits(uint x, uint k) { return (x << k) | (x >> (32u - k)); }

void hashFinal(inout uint a, inout uint b, inout uint c) {
    c ^= b;
    c -= rotateBits(b, 14u);
    a ^= c;
    a -= rotateBits(c, 11u);
    b ^= a;
    b -= rotateBits(a, 25u);
    c ^= b;
    c -= rotateBits(b, 16u);
    a ^= c;
    a -= rotateBits(c, 4u);
    b ^= a;
    b -= rotateBits(a, 14u);
    c ^= b;
    c -= rotateBits(b, 24u);
}

void hashMix(inout uint a, inout uint b, inout uint c) {
    a -= c;
    a ^= rotateBits(c, 4u);
    c += b;
    b -= a;
    b ^= rotateBits(a, 6u);
    a += c;
    c -= b;
    c ^= rotateBits(b, 8u);
    b += a;
    a -= c;
    a ^= rotateBits(c, 16u);
    c += b;
    b -= a;
    b ^= rotateBits(a, 19u);
    a += c;
    c -= b;
    c ^= rotateBits(b, 4u);
    b += a;
}

uint hashUint(uint kx) {
    uint a = 0xdeadbeefu + (1u << 2u) + 13u;
    uint b = a;
    uint c = a;
    a += kx;
    hashFinal(a, b, c);
    return c;
}

uint hashUint2(uint kx, uint ky) {
    uint a = 0xdeadbeefu + (2u << 2u) + 13u;
    uint b = a;
    uint c = a;
    b += ky;
    a += kx;
    hashFinal(a, b, c);
    return c;
}

uint hashUint3(uint kx, uint ky, uint kz) {
    uint a = 0xdeadbeefu + (3u << 2u) + 13u;
    uint b = a;
    uint c = a;
    c += kz;
    b += ky;
    a += kx;
    hashFinal(a, b, c);
    return c;
}

uint hashUint4(uint kx, uint ky, uint kz, uint kw) {
    uint a = 0xdeadbeefu + (4u << 2u) + 13u;
    uint b = a;
    uint c = a;
    a += kx;
    b += ky;
    c += kz;
    hashMix(a, b, c);
    a += kw;
    hashFinal(a, b, c);
    return c;
}

float hashToFloat(uint k) { return float(k) / 4294967295.0; }
float hashFloat(float k) { return hashToFloat(hashUint(asuint(k))); }
float hashFloat2(float2 k) { return hashToFloat(hashUint2(asuint(k.x), asuint(k.y))); }
float hashFloat3(float3 k) { return hashToFloat(hashUint3(asuint(k.x), asuint(k.y), asuint(k.z))); }
float hashFloat4(float4 k) { return hashToFloat(hashUint4(asuint(k.x), asuint(k.y), asuint(k.z), asuint(k.w))); }

//! A hash of the first dimensions components of k.
float hashFloatN(float4 k, uint dimensions) {
    switch (dimensions) {
    case 1: return hashFloat(k.x);
    case 2: return hashFloat2(k.xy);
    case 3: return hashFloat3(k.xyz);
    default: return hashFloat4(k);
    }
}

//! Blender's hash_floatN_to_float3.
float3 hashFloatNToFloat3(float4 k, uint dimensions) {
    switch (dimensions) {
    case 1: return float3(hashFloat(k.x), hashFloat2(float2(k.x, 1.0)), hashFloat2(float2(k.x, 2.0)));
    case 2: return float3(hashFloat2(k.xy), hashFloat3(float3(k.xy, 1.0)), hashFloat3(float3(k.xy, 2.0)));
    case 3: return float3(hashFloat3(k.xyz), hashFloat4(float4(k.xyz, 1.0)), hashFloat4(float4(k.xyz, 2.0)));
    default: return float3(hashFloat4(k), hashFloat4(k.zxwy), hashFloat4(k.wzyx));
    }
}

//! Blender's PCG hash of the cell of a Voronoi point, from Jarzynski and Olano 2020; the shift is arithmetic.
float4 hashCell(float4 cell, uint dimensions) {
    if (dimensions == 1)
        return float4(hashFloat(cell.x), hashFloat2(float2(cell.x, 1.0)), hashFloat2(float2(cell.x, 2.0)), 0.0);
    uint4 v = uint4(int4(cell)) * 1664525u + 1013904223u;
    [unroll]
    for (uint pass = 0; pass < 2; ++pass) {
        if (pass == 1)
            v ^= uint4(int4(v) >> 16);
        if (dimensions == 2) {
            v.x += v.y * 1664525u;
            v.y += v.x * 1664525u;
            continue;
        }
        v.x += v.y * (dimensions == 3 ? v.z : v.w);
        v.y += v.z * v.x;
        v.z += v.x * v.y;
        if (dimensions == 4)
            v.w += v.y * v.z;
    }
    return float4(v & 0x7fffffffu) / float(0x7fffffff);
}

//! The random offset of the point of a Voronoi cell; the components past dimensions are zero.
float4 cellPoint(float4 cell, uint dimensions) {
    const float4 hash = hashCell(cell, dimensions);
    return float4(hash.x, dimensions > 1 ? hash.y : 0.0, dimensions > 2 ? hash.z : 0.0, dimensions > 3 ? hash.w : 0.0);
}

float3 cellColor(float4 cell, uint dimensions) { return hashCell(dimensions == 2 ? float4(cell.xy, 0.0, 0.0) : cell, dimensions == 2 ? 3u : dimensions).xyz; }

float fade(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
float negateIf(float value, uint condition) { return condition != 0u ? -value : value; }

float gradient1(uint hash, float x) {
    const uint h = hash & 15u;
    const float g = 1.0 + float(h & 7u);
    return negateIf(g, h & 8u) * x;
}

float gradient2(uint hash, float x, float y) {
    const uint h = hash & 7u;
    const float u = h < 4u ? x : y;
    const float v = 2.0 * (h < 4u ? y : x);
    return negateIf(u, h & 1u) + negateIf(v, h & 2u);
}

float gradient3(uint hash, float x, float y, float z) {
    const uint h = hash & 15u;
    const float u = h < 8u ? x : y;
    const float vt = (h == 12u || h == 14u) ? x : z;
    const float v = h < 4u ? y : vt;
    return negateIf(u, h & 1u) + negateIf(v, h & 2u);
}

float gradient4(uint hash, float x, float y, float z, float w) {
    const uint h = hash & 31u;
    const float u = h < 24u ? x : y;
    const float v = h < 16u ? y : z;
    const float s = h < 8u ? z : w;
    return negateIf(u, h & 1u) + negateIf(v, h & 2u) + negateIf(s, h & 4u);
}

float perlin1(float p) {
    const float cell = floor(p);
    const float f = p - cell;
    const uint x = uint(int(cell));
    return lerp(gradient1(hashUint(x), f), gradient1(hashUint(x + 1u), f - 1.0), fade(f));
}

float perlin2(float2 p) {
    const float2 cell = floor(p);
    const float2 f = p - cell;
    const uint x = uint(int(cell.x));
    const uint y = uint(int(cell.y));
    const float u = fade(f.x);
    const float v = fade(f.y);
    const float a = lerp(gradient2(hashUint2(x, y), f.x, f.y), gradient2(hashUint2(x + 1u, y), f.x - 1.0, f.y), u);
    const float b = lerp(gradient2(hashUint2(x, y + 1u), f.x, f.y - 1.0), gradient2(hashUint2(x + 1u, y + 1u), f.x - 1.0, f.y - 1.0), u);
    return lerp(a, b, v);
}

float perlin3(float3 p) {
    const float3 cell = floor(p);
    const float3 f = p - cell;
    const uint x = uint(int(cell.x));
    const uint y = uint(int(cell.y));
    const uint z = uint(int(cell.z));
    const float3 t = float3(fade(f.x), fade(f.y), fade(f.z));
    float layers[2];
    [unroll]
    for (uint k = 0; k < 2; ++k) {
        const float fz = f.z - float(k);
        const float a = lerp(gradient3(hashUint3(x, y, z + k), f.x, f.y, fz), gradient3(hashUint3(x + 1u, y, z + k), f.x - 1.0, f.y, fz), t.x);
        const float b = lerp(gradient3(hashUint3(x, y + 1u, z + k), f.x, f.y - 1.0, fz), gradient3(hashUint3(x + 1u, y + 1u, z + k), f.x - 1.0, f.y - 1.0, fz), t.x);
        layers[k] = lerp(a, b, t.y);
    }
    return lerp(layers[0], layers[1], t.z);
}

float perlin4(float4 p) {
    const float4 cell = floor(p);
    const float4 f = p - cell;
    const uint4 c = uint4(int4(cell));
    const float4 t = float4(fade(f.x), fade(f.y), fade(f.z), fade(f.w));
    float volumes[2];
    [unroll]
    for (uint l = 0; l < 2; ++l) {
        float layers[2];
        [unroll]
        for (uint k = 0; k < 2; ++k) {
            const float fz = f.z - float(k);
            const float fw = f.w - float(l);
            const float a = lerp(gradient4(hashUint4(c.x, c.y, c.z + k, c.w + l), f.x, f.y, fz, fw),
                                 gradient4(hashUint4(c.x + 1u, c.y, c.z + k, c.w + l), f.x - 1.0, f.y, fz, fw), t.x);
            const float b = lerp(gradient4(hashUint4(c.x, c.y + 1u, c.z + k, c.w + l), f.x, f.y - 1.0, fz, fw),
                                 gradient4(hashUint4(c.x + 1u, c.y + 1u, c.z + k, c.w + l), f.x - 1.0, f.y - 1.0, fz, fw), t.x);
            layers[k] = lerp(a, b, t.y);
        }
        volumes[l] = lerp(layers[0], layers[1], t.z);
    }
    return lerp(volumes[0], volumes[1], t.w);
}

//! Blender's signed Perlin noise in the first dimensions components of p.
float signedNoise(float4 p, uint dimensions) {
    float value;
    switch (dimensions) {
    case 1: value = 0.25 * perlin1(p.x); break;
    case 2: value = 0.6616 * perlin2(p.xy); break;
    case 3: value = 0.982 * perlin3(p.xyz); break;
    default: value = 0.8344 * perlin4(p); break;
    }
    return isnan(value) || isinf(value) ? 0.0 : value;
}

//! Blender's random_floatN_offset; one dimension hashes the seed alone.
float4 noiseOffset(float seed, uint dimensions) {
    if (dimensions == 1)
        return float4(100.0 + hashFloat(seed) * 100.0, 0.0, 0.0, 0.0);
    return float4(100.0 + hashFloat2(float2(seed, 0.0)) * 100.0, 100.0 + hashFloat2(float2(seed, 1.0)) * 100.0, 100.0 + hashFloat2(float2(seed, 2.0)) * 100.0,
                  100.0 + hashFloat2(float2(seed, 3.0)) * 100.0);
}

static const uint noiseFbm = 0;
static const uint noiseMultifractal = 1;
static const uint noiseHybridMultifractal = 2;
static const uint noiseRidgedMultifractal = 3;
static const uint noiseHeteroTerrain = 4;

//! Blender's fractal noise types. settings is detail, roughness, lacunarity, normalize; shape is type, offset, gain.
float fractalNoise(float4 p, uint dimensions, float4 settings, float3 shape) {
    const float detail = clamp(settings.x, 0.0, 15.0);
    const float roughness = max(settings.y, 0.0);
    const float lacunarity = settings.z;
    const uint octaves = uint(detail);
    const float remainder = detail - floor(detail);
    const float offset = shape.y;
    const float gain = shape.z;
    switch (uint(shape.x)) {
    case noiseMultifractal: {
        float value = 1.0;
        float power = 1.0;
        [loop]
        for (uint i = 0; i <= octaves; ++i) {
            value *= power * signedNoise(p, dimensions) + 1.0;
            power *= roughness;
            p *= lacunarity;
        }
        if (remainder != 0.0)
            value *= remainder * power * signedNoise(p, dimensions) + 1.0;
        return value;
    }
    case noiseHybridMultifractal: {
        float power = 1.0;
        float value = 0.0;
        float weight = 1.0;
        [loop]
        for (uint i = 0; weight > 0.001 && i <= octaves; ++i) {
            weight = min(weight, 1.0);
            const float signal = (signedNoise(p, dimensions) + offset) * power;
            power *= roughness;
            value += weight * signal;
            weight *= gain * signal;
            p *= lacunarity;
        }
        if (remainder != 0.0 && weight > 0.001)
            value += remainder * min(weight, 1.0) * (signedNoise(p, dimensions) + offset) * power;
        return value;
    }
    case noiseRidgedMultifractal: {
        float power = roughness;
        float signal = offset - abs(signedNoise(p, dimensions));
        signal *= signal;
        float value = signal;
        [loop]
        for (uint i = 1; i <= octaves; ++i) {
            p *= lacunarity;
            const float weight = saturate(signal * gain);
            signal = offset - abs(signedNoise(p, dimensions));
            signal *= signal * weight;
            value += signal * power;
            power *= roughness;
        }
        return value;
    }
    case noiseHeteroTerrain: {
        float value = offset + signedNoise(p, dimensions);
        p *= lacunarity;
        float power = roughness;
        [loop]
        for (uint i = 1; i <= octaves; ++i) {
            value += (signedNoise(p, dimensions) + offset) * power * value;
            power *= roughness;
            p *= lacunarity;
        }
        if (remainder != 0.0)
            value += remainder * (signedNoise(p, dimensions) + offset) * power * value;
        return value;
    }
    default: {
        float scale = 1.0;
        float amplitude = 1.0;
        float maximum = 0.0;
        float sum = 0.0;
        [loop]
        for (uint i = 0; i <= octaves; ++i) {
            sum += signedNoise(scale * p, dimensions) * amplitude;
            maximum += amplitude;
            amplitude *= roughness;
            scale *= lacunarity;
        }
        const bool normalized = settings.w > 0.5;
        if (remainder == 0.0)
            return normalized ? 0.5 * sum / maximum + 0.5 : sum;
        const float last = sum + signedNoise(scale * p, dimensions) * amplitude;
        return normalized ? lerp(0.5 * sum / maximum + 0.5, 0.5 * last / (maximum + amplitude) + 0.5, remainder) : lerp(sum, last, remainder);
    }
    }
}

float4 maskDimensions(float4 p, uint dimensions) {
    return float4(p.x, dimensions > 1 ? p.y : 0.0, dimensions > 2 ? p.z : 0.0, dimensions > 3 ? p.w : 0.0);
}

//! Blender's Noise: coordinate vector and w; settings scale, detail, roughness, lacunarity; options distortion, normalize, dimensions, type, offset, gain.
[noinline]
float4 graphNoise(float4 coordinate, float4 settings, float4 options, float4 shape) {
    const uint dimensions = uint(options.z);
    float4 p = dimensions == 1 ? float4(coordinate.w, 0.0, 0.0, 0.0) : coordinate;
    p = maskDimensions(p, dimensions) * settings.x;
    const float distortion = options.x;
    if (distortion != 0.0) {
        float4 shift = float4(0.0, 0.0, 0.0, 0.0);
        [unroll]
        for (uint axis = 0; axis < 4; ++axis)
            if (axis < dimensions)
                shift[axis] = signedNoise(p + maskDimensions(noiseOffset(float(axis), dimensions), dimensions), dimensions) * distortion;
        p += shift;
    }
    const float3 kind = float3(options.w, shape.x, shape.y);
    const float4 parameters = float4(settings.y, settings.z, settings.w, options.y);
    const float value = fractalNoise(p, dimensions, parameters, kind);
    const float green = fractalNoise(p + maskDimensions(noiseOffset(float(dimensions), dimensions), dimensions), dimensions, parameters, kind);
    const float blue = fractalNoise(p + maskDimensions(noiseOffset(float(dimensions + 1), dimensions), dimensions), dimensions, parameters, kind);
    return float4(value, green, blue, value);
}

float voronoiDistance(float4 a, float4 b, uint metric, float exponent) {
    const float4 d = abs(a - b);
    switch (metric) {
    case 1: return d.x + d.y + d.z + d.w;
    case 2: return max(max(d.x, d.y), max(d.z, d.w));
    case 3: return pow(pow(d.x, exponent) + pow(d.y, exponent) + pow(d.z, exponent) + pow(d.w, exponent), 1.0 / exponent);
    default: return length(a - b);
    }
}

struct VoronoiOctave {
    float distance;
    float3 color;
    float4 position;
};

static const uint voronoiF1 = 0;
static const uint voronoiF2 = 1;
static const uint voronoiSmoothF1 = 2;
static const uint voronoiDistanceToEdge = 3;
static const uint voronoiSphereRadius = 4;

int voronoiReach(uint axis, uint dimensions, int reach) { return axis < dimensions ? reach : 0; }

//! One octave of F1, F2 or smooth F1 in cell space.
VoronoiOctave voronoiOctave(float4 coordinate, uint dimensions, uint feature, uint metric, float exponent, float randomness, float smoothness) {
    const float4 cell = floor(coordinate);
    const float4 local = coordinate - cell;
    const int reach = feature == voronoiSmoothF1 ? 2 : 1;
    const int rx = voronoiReach(0, dimensions, reach);
    const int ry = voronoiReach(1, dimensions, reach);
    const int rz = voronoiReach(2, dimensions, reach);
    const int rw = voronoiReach(3, dimensions, reach);
    const float smooth = clamp(smoothness * 0.5, 0.0, 0.5);
    VoronoiOctave first;
    first.distance = 8.0;
    first.color = float3(0.0, 0.0, 0.0);
    first.position = float4(0.0, 0.0, 0.0, 0.0);
    VoronoiOctave second = first;
    [loop]
    for (int w = -rw; w <= rw; ++w)
        [loop]
        for (int z = -rz; z <= rz; ++z)
            [loop]
            for (int y = -ry; y <= ry; ++y)
                [loop]
                for (int x = -rx; x <= rx; ++x) {
                    const float4 offset = float4(float(x), float(y), float(z), float(w));
                    const float4 point = offset + cellPoint(cell + offset, dimensions) * randomness;
                    const float distance = voronoiDistance(point, local, metric, exponent);
                    if (feature == voronoiSmoothF1) {
                        const float h = smoothstep(0.0, 1.0, 0.5 + 0.5 * (first.distance - distance) / max(smooth, 1e-8));
                        float correction = smooth * h * (1.0 - h);
                        first.distance = lerp(first.distance, distance, h) - correction;
                        correction /= 1.0 + 3.0 * smooth;
                        first.color = lerp(first.color, cellColor(cell + offset, dimensions), h) - correction;
                        first.position = lerp(first.position, point, h) - correction;
                        continue;
                    }
                    if (distance < first.distance) {
                        second = first;
                        first.distance = distance;
                        first.color = cellColor(cell + offset, dimensions);
                        first.position = point;
                    } else if (distance < second.distance) {
                        second.distance = distance;
                        second.color = cellColor(cell + offset, dimensions);
                        second.position = point;
                    }
                }
    VoronoiOctave result = feature == voronoiF2 ? second : first;
    result.position += cell;
    return result;
}

float voronoiEdge(float4 coordinate, uint dimensions, float randomness) {
    const float4 cell = floor(coordinate);
    const float4 local = coordinate - cell;
    const int rx = voronoiReach(0, dimensions, 1);
    const int ry = voronoiReach(1, dimensions, 1);
    const int rz = voronoiReach(2, dimensions, 1);
    const int rw = voronoiReach(3, dimensions, 1);
    float4 closest = float4(0.0, 0.0, 0.0, 0.0);
    float nearest = 8.0;
    [loop]
    for (int w = -rw; w <= rw; ++w)
        [loop]
        for (int z = -rz; z <= rz; ++z)
            [loop]
            for (int y = -ry; y <= ry; ++y)
                [loop]
                for (int x = -rx; x <= rx; ++x) {
                    const float4 offset = float4(float(x), float(y), float(z), float(w));
                    const float4 toPoint = offset + cellPoint(cell + offset, dimensions) * randomness - local;
                    const float distance = dot(toPoint, toPoint);
                    if (distance < nearest) {
                        nearest = distance;
                        closest = toPoint;
                    }
                }
    float edge = 8.0;
    [loop]
    for (int w = -rw; w <= rw; ++w)
        [loop]
        for (int z = -rz; z <= rz; ++z)
            [loop]
            for (int y = -ry; y <= ry; ++y)
                [loop]
                for (int x = -rx; x <= rx; ++x) {
                    const float4 offset = float4(float(x), float(y), float(z), float(w));
                    const float4 toPoint = offset + cellPoint(cell + offset, dimensions) * randomness - local;
                    const float4 perpendicular = toPoint - closest;
                    if (dot(perpendicular, perpendicular) > 0.0001)
                        edge = min(edge, dot((closest + toPoint) * 0.5, normalize(perpendicular)));
                }
    return edge;
}

float voronoiRadius(float4 coordinate, uint dimensions, float randomness) {
    const float4 cell = floor(coordinate);
    const float4 local = coordinate - cell;
    const int rx = voronoiReach(0, dimensions, 1);
    const int ry = voronoiReach(1, dimensions, 1);
    const int rz = voronoiReach(2, dimensions, 1);
    const int rw = voronoiReach(3, dimensions, 1);
    float4 closest = float4(0.0, 0.0, 0.0, 0.0);
    float4 closestOffset = float4(0.0, 0.0, 0.0, 0.0);
    float nearest = 8.0;
    [loop]
    for (int w = -rw; w <= rw; ++w)
        [loop]
        for (int z = -rz; z <= rz; ++z)
            [loop]
            for (int y = -ry; y <= ry; ++y)
                [loop]
                for (int x = -rx; x <= rx; ++x) {
                    const float4 offset = float4(float(x), float(y), float(z), float(w));
                    const float4 point = offset + cellPoint(cell + offset, dimensions) * randomness;
                    const float distance = length(point - local);
                    if (distance < nearest) {
                        nearest = distance;
                        closest = point;
                        closestOffset = offset;
                    }
                }
    nearest = 8.0;
    float4 neighbour = closest;
    [loop]
    for (int w = -rw; w <= rw; ++w)
        [loop]
        for (int z = -rz; z <= rz; ++z)
            [loop]
            for (int y = -ry; y <= ry; ++y)
                [loop]
                for (int x = -rx; x <= rx; ++x) {
                    if (x == 0 && y == 0 && z == 0 && w == 0)
                        continue;
                    const float4 offset = float4(float(x), float(y), float(z), float(w)) + closestOffset;
                    const float4 point = offset + cellPoint(cell + offset, dimensions) * randomness;
                    const float distance = length(closest - point);
                    if (distance < nearest) {
                        nearest = distance;
                        neighbour = point;
                    }
                }
    return length(neighbour - closest) * 0.5;
}

//! Blender's fractal Voronoi: coordinate vector and w; settings scale, detail, roughness, lacunarity; options randomness, exponent, smoothness, normalize.
VoronoiOctave voronoi(float4 coordinate, float4 settings, float4 options, float4 kind) {
    const uint dimensions = uint(kind.z);
    const uint feature = uint(kind.y);
    const uint metric = uint(kind.x);
    const float randomness = saturate(options.x);
    float4 p = dimensions == 1 ? float4(coordinate.w, 0.0, 0.0, 0.0) : coordinate;
    p = maskDimensions(p, dimensions) * settings.x;
    VoronoiOctave result;
    result.distance = 0.0;
    result.color = float3(0.0, 0.0, 0.0);
    result.position = float4(0.0, 0.0, 0.0, 0.0);
    if (feature == voronoiDistanceToEdge) {
        result.distance = voronoiEdge(p, dimensions, randomness);
        return result;
    }
    if (feature == voronoiSphereRadius) {
        result.distance = voronoiRadius(p, dimensions, randomness);
        return result;
    }
    const float detail = clamp(settings.y, 0.0, 15.0);
    const float roughness = saturate(settings.z);
    const float lacunarity = settings.w;
    const bool single = detail == 0.0 || roughness == 0.0;
    float amplitude = 1.0;
    float maximum = 0.0;
    float scale = 1.0;
    [loop]
    for (uint i = 0; i <= uint(ceil(detail)); ++i) {
        const VoronoiOctave octave = voronoiOctave(p * scale, dimensions, feature, metric, options.y, randomness, options.z);
        if (single) {
            maximum = 1.0;
            result = octave;
            break;
        }
        if (float(i) <= detail) {
            maximum += amplitude;
            result.distance += octave.distance * amplitude;
            result.color += octave.color * amplitude;
            result.position = lerp(result.position, octave.position / scale, amplitude);
            scale *= lacunarity;
            amplitude *= roughness;
            continue;
        }
        const float remainder = detail - floor(detail);
        if (remainder != 0.0) {
            maximum = lerp(maximum, maximum + amplitude, remainder);
            result.distance = lerp(result.distance, result.distance + octave.distance * amplitude, remainder);
            result.color = lerp(result.color, result.color + octave.color * amplitude, remainder);
            result.position = lerp(result.position, lerp(result.position, octave.position / scale, amplitude), remainder);
        }
    }
    if (options.w > 0.5) {
        const float diagonal = voronoiDistance(maskDimensions(float4(1.0, 1.0, 1.0, 1.0), dimensions), float4(0.0, 0.0, 0.0, 0.0), metric, options.y);
        const float reach = (0.5 + 0.5 * randomness) * diagonal * (feature == voronoiF2 ? 2.0 : 1.0);
        result.distance /= maximum * reach;
        result.color /= maximum;
    }
    result.position = settings.x != 0.0 ? result.position / settings.x : float4(0.0, 0.0, 0.0, 0.0);
    if (dimensions == 1)
        result.position = float4(0.0, 0.0, 0.0, result.position.x);
    return result;
}

//! The colour and the distance, or the radius for the n-sphere feature.
[noinline]
float4 graphVoronoi(float4 coordinate, float4 settings, float4 options, float4 kind) {
    const VoronoiOctave result = voronoi(coordinate, settings, options, kind);
    return float4(result.color, result.distance);
}

//! The position and w.
[noinline]
float4 graphVoronoiPosition(float4 coordinate, float4 settings, float4 options, float4 kind) { return voronoi(coordinate, settings, options, kind).position; }

//! Blender's Wave Texture. settings is scale, distortion, detail, detail scale; options is detail roughness, phase, type (0 bands, 1 rings), direction.
[noinline]
float graphWave(float3 coordinate, float4 settings, float4 options, float profile) {
    const float3 p = (coordinate * settings.x + 0.000001) * 0.999999;
    const uint direction = uint(options.w);
    float n;
    if (options.z < 0.5) {
        n = direction == 3 ? (p.x + p.y + p.z) * 10.0 : p[direction] * 20.0;
    } else {
        float3 ring = p;
        if (direction < 3)
            ring[direction] = 0.0;
        n = length(ring) * 20.0;
    }
    n += options.y;
    if (settings.y != 0.0)
        n += settings.y * (fractalNoise(float4(p * settings.w, 0.0), 3, float4(settings.z, options.x, 2.0, 1.0), float3(0.0, 0.0, 0.0)) * 2.0 - 1.0);
    switch (uint(profile)) {
    case 1: {
        const float saw = n / twoPi;
        return saw - floor(saw);
    }
    case 2: {
        const float triangle = n / twoPi;
        return abs(triangle - floor(triangle + 0.5)) * 2.0;
    }
    default:
        return 0.5 + 0.5 * sin(n - 0.5 * pi);
    }
}

//! Blender's Magic Texture. settings is scale, distortion, depth. The result is the colour and its average.
[noinline]
float4 graphMagic(float3 coordinate, float4 settings) {
    const float3 p = coordinate * settings.x;
    float distortion = settings.y;
    const int depth = int(settings.z);
    float x = sin((p.x + p.y + p.z) * 5.0);
    float y = cos((-p.x + p.y - p.z) * 5.0);
    float z = -cos((-p.x - p.y + p.z) * 5.0);
    if (depth > 0) {
        x *= distortion;
        y *= distortion;
        z *= distortion;
        y = -cos(x - y + z) * distortion;
        if (depth > 1) {
            x = cos(x - y - z) * distortion;
            if (depth > 2) {
                z = sin(-x - y - z) * distortion;
                if (depth > 3) {
                    x = -cos(-x + y - z) * distortion;
                    if (depth > 4) {
                        y = -sin(-x + y + z) * distortion;
                        if (depth > 5) {
                            y = -cos(-x + y + z) * distortion;
                            if (depth > 6) {
                                x = cos(x + y + z) * distortion;
                                if (depth > 7) {
                                    z = sin(x + y - z) * distortion;
                                    if (depth > 8) {
                                        x = -cos(-x - y + z) * distortion;
                                        if (depth > 9)
                                            y = -sin(x - y + z) * distortion;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (distortion != 0.0) {
        distortion *= 2.0;
        x /= distortion;
        y /= distortion;
        z /= distortion;
    }
    const float3 color = float3(0.5 - x, 0.5 - y, 0.5 - z);
    return float4(color, (color.x + color.y + color.z) / 3.0);
}

//! Blender's Checker Texture factor: 1 where the first colour shows.
[noinline]
float graphChecker(float3 coordinate, float scale) {
    const float3 p = (coordinate * scale + 0.000001) * 0.999999;
    const int3 cell = abs(int3(floor(p)));
    return ((cell.x % 2 == cell.y % 2) == (cell.z % 2 != 0)) ? 1.0 : 0.0;
}

float brickNoise(uint n) {
    n = (n + 1013u) & 0x7fffffffu;
    n = (n >> 13u) ^ n;
    const uint nn = (n * (n * n * 60493u + 19990303u) + 1376312589u) & 0x7fffffffu;
    return 0.5 * (float(nn) / 1073741824.0);
}

//! Blender's Brick: offsets offset and squash with frequencies; settings scale, mortar size and smoothness, bias; size brick width, row height.
[noinline]
float2 graphBrick(float3 coordinate, float4 offsets, float4 settings, float2 size) {
    const float3 p = coordinate * settings.x;
    float brickWidth = size.x;
    const float rowHeight = size.y;
    const int row = int(floor(p.y / rowHeight));
    const int offsetFrequency = int(offsets.y);
    const int squashFrequency = int(offsets.w);
    float offset = 0.0;
    if (offsetFrequency != 0 && squashFrequency != 0) {
        brickWidth *= (row % squashFrequency) != 0 ? 1.0 : offsets.z;
        offset = (row % offsetFrequency) != 0 ? 0.0 : brickWidth * offsets.x;
    }
    const int brick = int(floor((p.x + offset) / brickWidth));
    const float x = (p.x + offset) - brickWidth * float(brick);
    const float y = p.y - rowHeight * float(row);
    const float tint = saturate(brickNoise(uint((row << 16) + (brick & 0xffff))) + settings.w);
    float edge = min(min(x, y), min(brickWidth - x, rowHeight - y));
    const float mortarSize = settings.y;
    const float mortarSmooth = settings.z;
    float mortar;
    if (edge >= mortarSize)
        mortar = 0.0;
    else if (mortarSmooth == 0.0)
        mortar = 1.0;
    else {
        edge = 1.0 - edge / mortarSize;
        mortar = edge < mortarSmooth ? smoothstep(0.0, 1.0, edge / mortarSmooth) : 1.0;
    }
    return float2(tint, mortar);
}

//! Blender's Gradient Texture: linear, quadratic, easing, diagonal, radial, quadratic sphere, spherical.
[noinline]
float graphGradient(float3 p, float kind) {
    float value;
    switch (uint(kind)) {
    case 1: {
        const float r = max(p.x, 0.0);
        value = r * r;
        break;
    }
    case 2: {
        const float r = saturate(p.x);
        const float t = r * r;
        value = 3.0 * t - 2.0 * t * r;
        break;
    }
    case 3: value = (p.x + p.y) * 0.5; break;
    case 4: value = atan2(p.y, p.x) / twoPi + 0.5; break;
    case 5: {
        const float r = max(0.999999 - length(p), 0.0);
        value = r * r;
        break;
    }
    case 6: value = max(0.999999 - length(p), 0.0); break;
    default: value = p.x; break;
    }
    return saturate(value);
}

//! Blender's White Noise Texture: the colour and the value.
[noinline]
float4 graphWhiteNoise(float4 coordinate, float dimensions) {
    const uint count = uint(dimensions);
    const float4 p = count == 1 ? float4(coordinate.w, 0.0, 0.0, 0.0) : coordinate;
    return float4(hashFloatNToFloat3(p, count), hashFloatN(p, count));
}

static const uint gaborImpulses = 8u;

float2 polarToCartesian(float radius, float angle) {
    return radius * float2(cos(angle), sin(angle));
}

float2 gaborKernel(float distanceSquared, float angle) {
    const float envelope = exp(-pi * distanceSquared) * (0.5 + 0.5 * cos(pi * distanceSquared));
    return polarToCartesian(envelope, angle);
}

float2 gaborCell2(float2 cell, float2 position, float frequency, float isotropy, float baseOrientation) {
    float2 noise = float2(0.0, 0.0);
    for (uint impulse = 0; impulse < gaborImpulses; ++impulse) {
        const float seed = float(impulse * 3u);
        const float orientation = baseOrientation + (hashFloat3(float3(cell, seed)) - 0.5) * pi * isotropy;
        const float3 centerSeed = float3(cell, seed + 1.0);
        const float2 offset = position - float2(hashFloat3(centerSeed), hashFloat3(centerSeed.zxy));
        const float distanceSquared = dot(offset, offset);
        if (distanceSquared >= 1.0)
            continue;
        const float weight = hashFloat3(float3(cell, seed + 2.0)) < 0.5 ? -1.0 : 1.0;
        noise += weight * gaborKernel(distanceSquared, 2.0 * pi * dot(offset, polarToCartesian(frequency, orientation)));
    }
    return noise;
}

float3 gaborOrientation(float3 orientation, float isotropy, float4 seed) {
    if (isotropy == 0.0)
        return orientation;
    const float2 angles = float2(hashFloat4(seed), hashFloat4(seed.zxwy)) * pi;
    const float inclination = acos(orientation.z) + angles.x * isotropy;
    const float azimuth = (orientation.y < 0.0 ? -1.0 : 1.0) * acos(orientation.x / length(orientation.xy)) + angles.y * isotropy;
    return float3(sin(inclination) * cos(azimuth), sin(inclination) * sin(azimuth), cos(inclination));
}

float2 gaborCell3(float3 cell, float3 position, float frequency, float isotropy, float3 baseOrientation) {
    float2 noise = float2(0.0, 0.0);
    for (uint impulse = 0; impulse < gaborImpulses; ++impulse) {
        const float seed = float(impulse * 3u);
        const float3 orientation = gaborOrientation(baseOrientation, isotropy, float4(cell, seed));
        const float4 centerSeed = float4(cell, seed + 1.0);
        const float3 offset = position - float3(hashFloat4(centerSeed), hashFloat4(centerSeed.zxwy), hashFloat4(centerSeed.wzyx));
        const float distanceSquared = dot(offset, offset);
        if (distanceSquared >= 1.0)
            continue;
        const float weight = hashFloat4(float4(cell, seed + 2.0)) < 0.5 ? -1.0 : 1.0;
        noise += weight * gaborKernel(distanceSquared, 2.0 * pi * dot(offset, frequency * orientation));
    }
    return noise;
}

//! Cycles' svm_node_tex_gabor: settings scale, frequency, anisotropy and 2D orientation; the result is the value, phase and intensity.
[noinline]
float3 graphGabor(float3 coordinate, float4 settings, float3 orientation, float dimensions) {
    const float3 scaled = coordinate * settings.x;
    const float frequency = max(settings.y, 0.001);
    const float isotropy = 1.0 - saturate(settings.z);
    float2 phasor = float2(0.0, 0.0);
    float deviation = 0.0;
    if (dimensions < 2.5) {
        const float2 cell = floor(scaled.xy);
        const float2 local = scaled.xy - cell;
        for (int j = -1; j <= 1; ++j)
            for (int i = -1; i <= 1; ++i)
                phasor += gaborCell2(cell + float2(i, j), local - float2(i, j), frequency, isotropy, settings.w);
        deviation = sqrt(float(gaborImpulses) * 0.5 * 0.25);
    } else {
        const float3 direction = normalize(orientation);
        const float3 cell = floor(scaled);
        const float3 local = scaled - cell;
        for (int k = -1; k <= 1; ++k)
            for (int j = -1; j <= 1; ++j)
                for (int i = -1; i <= 1; ++i)
                    phasor += gaborCell3(cell + float3(i, j, k), local - float3(i, j, k), frequency, isotropy, direction);
        deviation = sqrt(float(gaborImpulses) * 0.5 / (4.0 * sqrt(2.0)));
    }
    const float normalization = 6.0 * deviation;
    return float3(phasor.y / normalization * 0.5 + 0.5, (atan2(phasor.y, phasor.x) + pi) / (2.0 * pi), length(phasor) / normalization);
}

//! Cycles' weights of the three sides in a box projection: the normal is in object space, and blend widens the zones where sides mix.
[noinline]
float3 graphBoxWeights(float3 normal, float blend) {
    float3 n = abs(normal);
    n /= n.x + n.y + n.z;
    const float limit = 0.5 * (1.0 + blend);
    if (n.x > limit * (n.x + n.y) && n.x > limit * (n.x + n.z))
        return float3(1.0, 0.0, 0.0);
    if (n.y > limit * (n.x + n.y) && n.y > limit * (n.y + n.z))
        return float3(0.0, 1.0, 0.0);
    if (n.z > limit * (n.x + n.z) && n.z > limit * (n.y + n.z))
        return float3(0.0, 0.0, 1.0);
    if (blend <= 0.0)
        return float3(1.0, 0.0, 0.0);
    if (n.z < (1.0 - limit) * (n.y + n.x)) {
        const float x = saturate((n.x / (n.x + n.y) - 0.5 * (1.0 - blend)) / blend);
        return float3(x, 1.0 - x, 0.0);
    }
    if (n.x < (1.0 - limit) * (n.y + n.z)) {
        const float y = saturate((n.y / (n.y + n.z) - 0.5 * (1.0 - blend)) / blend);
        return float3(0.0, y, 1.0 - y);
    }
    if (n.y < (1.0 - limit) * (n.x + n.z)) {
        const float x = saturate((n.x / (n.x + n.z) - 0.5 * (1.0 - blend)) / blend);
        return float3(x, 0.0, 1.0 - x);
    }
    return ((2.0 - limit) * n + (limit - 1.0)) / (2.0 * limit - 1.0);
}

float3 rgbToHsv(float3 rgb) {
    const float high = max(rgb.x, max(rgb.y, rgb.z));
    const float low = min(rgb.x, min(rgb.y, rgb.z));
    const float delta = high - low;
    const float saturation = high != 0.0 ? delta / high : 0.0;
    float hue = 0.0;
    if (saturation != 0.0) {
        const float3 c = (float3(high, high, high) - rgb) / delta;
        if (rgb.x == high)
            hue = c.z - c.y;
        else if (rgb.y == high)
            hue = 2.0 + c.x - c.z;
        else
            hue = 4.0 + c.y - c.x;
        hue /= 6.0;
        if (hue < 0.0)
            hue += 1.0;
    }
    return float3(hue, saturation, high);
}

float3 hsvToRgb(float3 hsv) {
    const float s = hsv.y;
    const float v = hsv.z;
    if (s == 0.0)
        return float3(v, v, v);
    float h = hsv.x == 1.0 ? 0.0 : hsv.x;
    h *= 6.0;
    const float i = floor(h);
    const float f = h - i;
    const float p = v * (1.0 - s);
    const float q = v * (1.0 - s * f);
    const float t = v * (1.0 - s * (1.0 - f));
    switch (int(i)) {
    case 0: return float3(v, t, p);
    case 1: return float3(q, v, p);
    case 2: return float3(p, v, t);
    case 3: return float3(p, q, v);
    case 4: return float3(t, p, v);
    default: return float3(v, p, q);
    }
}

[noinline]
float3 graphRgbToHsv(float3 rgb) { return rgbToHsv(rgb); }

[noinline]
float3 graphHsvToRgb(float3 hsv) { return hsvToRgb(hsv); }

//! Blender's Hue/Saturation/Value node. settings is hue, saturation, value, factor.
[noinline]
float3 graphHueSaturationValue(float3 color, float4 settings) {
    float3 hsv = rgbToHsv(color);
    hsv.x = frac(hsv.x + settings.x + 0.5);
    hsv.y = saturate(hsv.y * settings.y);
    hsv.z *= settings.z;
    return max(lerp(color, hsvToRgb(hsv), settings.w), 0.0);
}

float3 xyzToLinearSrgb(float3 xyz) {
    return float3(3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z, -0.969266 * xyz.x + 1.8760108 * xyz.y + 0.041556 * xyz.z,
                  0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z);
}

//! The Planckian locus fit of Kim et al. 2002 in linear sRGB, scaled to a luminance of 1 as the Blackbody of Cycles.
[noinline]
float3 graphBlackbody(float temperature) {
    const float t = clamp(temperature, 1667.0, 25000.0);
    const float u = 1000.0 / t;
    const float x = t <= 4000.0 ? ((-0.2661239 * u - 0.2343589) * u + 0.8776956) * u + 0.17991
                                : ((-3.0258469 * u + 2.1070379) * u + 0.2226347) * u + 0.24039;
    float y;
    if (t <= 2222.0)
        y = ((-1.1063814 * x - 1.3481102) * x + 2.18555832) * x - 0.20219683;
    else if (t <= 4000.0)
        y = ((-0.9549476 * x - 1.37418593) * x + 2.09137015) * x - 0.16748867;
    else
        y = ((3.081758 * x - 5.8733867) * x + 3.75112997) * x - 0.37001483;
    const float3 rgb = max(xyzToLinearSrgb(float3(x / y, 1.0, (1.0 - x - y) / y)), 0.0);
    return rgb / max(luminance(rgb), 1e-6);
}

float lobe(float x, float mean, float below, float above) {
    const float t = (x - mean) / (x < mean ? below : above);
    return exp(-0.5 * t * t);
}

//! The CIE 1931 fit of Wyman et al. 2013 in linear sRGB, scaled as Blender's Wavelength node.
[noinline]
float3 graphWavelength(float wavelength) {
    const float x = 1.056 * lobe(wavelength, 599.8, 37.9, 31.0) + 0.362 * lobe(wavelength, 442.0, 16.0, 26.7) - 0.065 * lobe(wavelength, 501.1, 20.4, 26.2);
    const float y = 0.821 * lobe(wavelength, 568.8, 46.9, 40.5) + 0.286 * lobe(wavelength, 530.9, 16.3, 31.1);
    const float z = 1.217 * lobe(wavelength, 437.0, 11.8, 36.0) + 0.681 * lobe(wavelength, 459.0, 26.0, 13.8);
    return max(xyzToLinearSrgb(float3(x, y, z)) / 2.52, 0.0);
}

float fresnelDielectricCosine(float cosine, float eta) {
    const float c = abs(cosine);
    float g = eta * eta - 1.0 + c * c;
    if (g <= 0.0)
        return 1.0;
    g = sqrt(g);
    const float a = (g - c) / (g + c);
    const float b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * a * a * (1.0 + b * b);
}

//! Blender's Fresnel node; incoming points to the viewer.
[noinline]
float graphFresnel(float ior, float3 normal, float3 incoming, float backfacing) {
    float eta = max(ior, 1e-5);
    eta = backfacing > 0.5 ? 1.0 / eta : eta;
    return fresnelDielectricCosine(dot(incoming, normal), eta);
}

//! Blender's Layer Weight node: the fresnel and the facing factor.
[noinline]
float2 graphLayerWeight(float blend, float3 normal, float3 incoming, float backfacing) {
    float eta = max(1.0 - blend, 1e-5);
    eta = backfacing > 0.5 ? eta : 1.0 / eta;
    const float fresnel = fresnelDielectricCosine(dot(incoming, normal), eta);
    float facing = abs(dot(incoming, normal));
    if (blend != 0.5) {
        float exponent = clamp(blend, 0.0, 1.0 - 1e-5);
        exponent = exponent < 0.5 ? 2.0 * exponent : 0.5 / (1.0 - exponent);
        facing = pow(facing, exponent);
    }
    return float2(fresnel, 1.0 - facing);
}

MaterialGraphOutput neutralGraphOutput(ShadingContext context) {
    MaterialGraphOutput output;
    output.baseColorAlpha = float4(0.8, 0.8, 0.8, 1.0);
    output.metallicRoughnessIorSpecular = float4(0.0, 0.5, 1.5, 0.5);
    output.specularTintAnisotropic = float4(1.0, 1.0, 1.0, 0.0);
    output.normalRotation = float4(context.normal.xyz, 0.0);
    output.tangentTransmission = float4(context.tangent.xyz, 0.0);
    output.coat = float4(0.0, 0.03, 1.5, 0.0);
    output.coatTint = float4(1.0, 1.0, 1.0, 0.0);
    output.coatNormal = float4(context.normal.xyz, 0.0);
    output.sheen = float4(0.0, 0.5, 0.0, 0.0);
    output.sheenTint = float4(1.0, 1.0, 1.0, 0.0);
    output.emission = float4(0.0, 0.0, 0.0, 0.0);
    output.thinFilmDispersion = float4(0.0, 1.33, 0.0, 0.0);
    output.closure = float4(1.0, 1.0, 1.0, 0.0);
    output.subsurface = float4(0.0, 0.0, 0.0, 0.0);
    output.transparency = 0.0;
    return output;
}

//! A run of an alpha graph for a ray that passes a surface: the context, the material data, the hit for the trace helpers, and the outputs.
struct MaterialGraphCall {
    ShadingContext context;
    uint4 material;
    Ptr<SceneRecord> scene;
    uint instance;
    uint primitive;
    uint seed;
    MaterialGraphOutput output;
};

//! The pipeline cache replaces this body with a switch that runs alpha graph index of the AlphaGraph kernel; there is no such kernel without the link.
[noinline]
MaterialGraphCall runAlphaGraph(uint index, MaterialGraphCall call) { return call; }

//! The pipeline cache replaces this body per material kernel; a branch on an impossible address keeps every helper in the module.
[noinline]
MaterialGraphOutput evaluateMaterialGraph(ShadingContext context, uint4 material) {
    MaterialGraphOutput output = neutralGraphOutput(context);
    if (material.x != 0xffffffffu)
        return output;
    const float4 p = context.position;
    output.baseColorAlpha = graphParameter(material, p.x) + graphTexture(material, context.uvGradients, p.y, context.uv01.xy) + graphNoise(p, p, p, p) + graphVoronoi(p, p, p, p) +
                            graphVoronoiPosition(p, p, p, p) + graphMagic(p.xyz, p) + graphWhiteNoise(p, p.w);
    output.metallicRoughnessIorSpecular = float4(graphWave(p.xyz, p, p, p.w), graphChecker(p.xyz, p.w), graphBrick(p.xyz, p, p, p.xy));
    output.specularTintAnisotropic = float4(graphHueSaturationValue(p.xyz, p) + graphRgbToHsv(p.xyz) + graphHsvToRgb(p.xyz), graphGradient(p.xyz, p.w));
    output.coatTint = float4(graphBlackbody(p.x) + graphWavelength(p.y), graphFresnel(p.z, p.xyz, p.xyz, p.w));
    output.sheen = float4(graphLayerWeight(p.x, p.xyz, p.xyz, p.w), graphAo(p.xyz, p.xyz, p), graphRadialTilingSegment(p.xyz, p.w));
    output.sheenTint = float4(graphBevel(p.xyz, p.xyz, p.xyz, p) + graphRaycast(p.xyz, p.xyz, p) + graphRaycastNormal(p.xyz, p.xyz, p),
                              graphGabor(p.xyz, p, p.xyz, p.w).x + graphRadialTiling(p.xyz, p).x + graphBoxWeights(p.xyz, p.w).x + graphInstanceAttribute(p.x).x + graphParametric(p.xyz).x + graphWireframe(p.xyz, p.w));
    return output;
}
