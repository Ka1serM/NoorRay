#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/Shared.h"

NR_NAMESPACE_BEGIN

static const uint maxProgramRegisters = 64u;

static const uint programOpMath = 0u;
static const uint programOpVectorMath = 1u;
static const uint programOpWave = 2u;
static const uint programOpCombine = 3u;
static const uint programOpSeparate = 4u;
static const uint programOpAverage = 5u;
static const uint programOpGrayscale = 6u;
static const uint programOpPosition = 7u;
static const uint programOpNormal = 8u;
static const uint programOpGenerated = 9u;
static const uint programOpAttribute = 10u;
static const uint programOpDisplacement = 11u;
static const uint programOpVectorDisplacement = 12u;

static const uint mathAdd = 0u;
static const uint mathSubtract = 1u;
static const uint mathMultiply = 2u;
static const uint mathDivide = 3u;
static const uint mathMultiplyAdd = 4u;
static const uint mathPower = 5u;
static const uint mathLogarithm = 6u;
static const uint mathSqrt = 7u;
static const uint mathInverseSqrt = 8u;
static const uint mathAbsolute = 9u;
static const uint mathExponent = 10u;
static const uint mathMinimum = 11u;
static const uint mathMaximum = 12u;
static const uint mathLessThan = 13u;
static const uint mathGreaterThan = 14u;
static const uint mathSign = 15u;
static const uint mathCompare = 16u;
static const uint mathSmoothMin = 17u;
static const uint mathSmoothMax = 18u;
static const uint mathRound = 19u;
static const uint mathFloor = 20u;
static const uint mathCeil = 21u;
static const uint mathTruncate = 22u;
static const uint mathFraction = 23u;
static const uint mathTruncatedModulo = 24u;
static const uint mathFlooredModulo = 25u;
static const uint mathWrap = 26u;
static const uint mathSnap = 27u;
static const uint mathPingPong = 28u;
static const uint mathSine = 29u;
static const uint mathCosine = 30u;
static const uint mathTangent = 31u;
static const uint mathArcsine = 32u;
static const uint mathArccosine = 33u;
static const uint mathArctangent = 34u;
static const uint mathArctan2 = 35u;
static const uint mathSinh = 36u;
static const uint mathCosh = 37u;
static const uint mathTanh = 38u;
static const uint mathRadians = 39u;
static const uint mathDegrees = 40u;
static const uint mathClamp = 128u;

static const uint vectorMathAdd = 0u;
static const uint vectorMathSubtract = 1u;
static const uint vectorMathMultiply = 2u;
static const uint vectorMathDivide = 3u;
static const uint vectorMathMultiplyAdd = 4u;
static const uint vectorMathCrossProduct = 5u;
static const uint vectorMathProject = 6u;
static const uint vectorMathReflect = 7u;
static const uint vectorMathRefract = 8u;
static const uint vectorMathFaceforward = 9u;
static const uint vectorMathDotProduct = 10u;
static const uint vectorMathDistance = 11u;
static const uint vectorMathLength = 12u;
static const uint vectorMathScale = 13u;
static const uint vectorMathNormalize = 14u;
static const uint vectorMathAbsolute = 15u;
static const uint vectorMathPower = 16u;
static const uint vectorMathSign = 17u;
static const uint vectorMathMinimum = 18u;
static const uint vectorMathMaximum = 19u;
static const uint vectorMathFloor = 20u;
static const uint vectorMathCeil = 21u;
static const uint vectorMathFraction = 22u;
static const uint vectorMathModulo = 23u;
static const uint vectorMathWrap = 24u;
static const uint vectorMathSnap = 25u;
static const uint vectorMathSine = 26u;
static const uint vectorMathCosine = 27u;
static const uint vectorMathTangent = 28u;

static const uint waveRings = 1u;
static const uint waveDirectionX = 0u;
static const uint waveDirectionY = 1u;
static const uint waveDirectionZ = 2u;
static const uint waveProfileSaw = 1u;
static const uint waveProfileTriangle = 2u;

static const float programPi = 3.14159265358979f;

NR_SHARED uint getProgramOp(uint2 instruction) {
    return instruction.x & 0xffu;
}
NR_SHARED uint getProgramMode(uint2 instruction) {
    return (instruction.x >> 8u) & 0xffu;
}
NR_SHARED uint getProgramOutput(uint2 instruction) {
    return (instruction.x >> 16u) & 0xffu;
}
NR_SHARED uint getProgramInput(uint2 instruction, uint input) {
    return input == 0u ? instruction.x >> 24u : (instruction.y >> ((input - 1u) * 8u)) & 0xffu;
}

NR_SHARED float4 broadcast(float value) {
    return float4(value, value, value, value);
}
NR_SHARED float4 fromVector(float3 value) {
    return float4(value.x, value.y, value.z, 0.0f);
}
NR_SHARED float3 toVector(float4 value) {
    return float3(value.x, value.y, value.z);
}

NR_SHARED uint rotateBits(uint value, uint count) {
    return (value << count) | (value >> (32u - count));
}

//! Bob Jenkins' lookup3 final mix, as Blender's hash_uint3.
NR_SHARED uint hashLattice(uint x, uint y, uint z) {
    uint a = 0xdeadbeefu + (3u << 2u) + 13u;
    uint b = a;
    uint c = a;
    c += z;
    b += y;
    a += x;
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
    return c;
}

NR_SHARED float negateIf(float value, uint condition) {
    return condition != 0u ? -value : value;
}

NR_SHARED float latticeGradient(uint hash, float x, float y, float z) {
    const uint h = hash & 15u;
    const float u = h < 8u ? x : y;
    const float other = (h == 12u || h == 14u) ? x : z;
    const float v = h < 4u ? y : other;
    return negateIf(u, h & 1u) + negateIf(v, h & 2u);
}

NR_SHARED float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

NR_SHARED float perlinCorner(int x, int y, int z, float fx, float fy, float fz) {
    return latticeGradient(hashLattice(uint(x), uint(y), uint(z)), fx, fy, fz);
}

NR_SHARED float perlinNoise(float3 p) {
    const int x = int(floor(p.x));
    const int y = int(floor(p.y));
    const int z = int(floor(p.z));
    const float fx = p.x - floor(p.x);
    const float fy = p.y - floor(p.y);
    const float fz = p.z - floor(p.z);
    const float u = fade(fx);
    const float v = fade(fy);
    const float w = fade(fz);
    const float c000 = perlinCorner(x, y, z, fx, fy, fz);
    const float c100 = perlinCorner(x + 1, y, z, fx - 1.0f, fy, fz);
    const float c010 = perlinCorner(x, y + 1, z, fx, fy - 1.0f, fz);
    const float c110 = perlinCorner(x + 1, y + 1, z, fx - 1.0f, fy - 1.0f, fz);
    const float c001 = perlinCorner(x, y, z + 1, fx, fy, fz - 1.0f);
    const float c101 = perlinCorner(x + 1, y, z + 1, fx - 1.0f, fy, fz - 1.0f);
    const float c011 = perlinCorner(x, y + 1, z + 1, fx, fy - 1.0f, fz - 1.0f);
    const float c111 = perlinCorner(x + 1, y + 1, z + 1, fx - 1.0f, fy - 1.0f, fz - 1.0f);
    const float lower = (1.0f - v) * (c000 * (1.0f - u) + c100 * u) + v * (c010 * (1.0f - u) + c110 * u);
    const float upper = (1.0f - v) * (c001 * (1.0f - u) + c101 * u) + v * (c011 * (1.0f - u) + c111 * u);
    //! Blender scales its 3D Perlin noise by 0.982.
    return 0.982f * ((1.0f - w) * lower + w * upper);
}

NR_SHARED float fractalNoise(float3 p, float detail, float roughness) {
    const float octaves = clamp(detail, 0.0f, 15.0f);
    const int whole = int(octaves);
    float scale = 1.0f;
    float amplitude = 1.0f;
    float total = 0.0f;
    float sum = 0.0f;
    for (int octave = 0; octave <= whole; octave++) {
        sum += perlinNoise(p * scale) * amplitude;
        total += amplitude;
        amplitude *= roughness;
        scale *= 2.0f;
    }
    const float remainder = octaves - floor(octaves);
    const float value = 0.5f * sum / total + 0.5f;
    if (remainder == 0.0f)
        return value;
    const float more = sum + perlinNoise(p * scale) * amplitude;
    return lerp(value, 0.5f * more / (total + amplitude) + 0.5f, remainder);
}

//! settings holds scale, distortion, detail and detail scale; shape holds detail roughness and phase offset.
NR_SHARED float waveTexture(uint mode, float3 vector, float4 settings, float4 shape) {
    const float scale = settings.x;
    const float distortion = settings.y;
    //! Blender nudges the coordinates to avoid precision issues on unit coordinates.
    const float3 p = (vector * scale + float3(0.000001f, 0.000001f, 0.000001f)) * 0.999999f;
    const uint direction = (mode >> 1u) & 3u;
    float n = 0.0f;
    if ((mode & 1u) != waveRings) {
        n = direction == waveDirectionX   ? p.x * 20.0f
            : direction == waveDirectionY ? p.y * 20.0f
            : direction == waveDirectionZ ? p.z * 20.0f
                                          : (p.x + p.y + p.z) * 10.0f;
    } else {
        const float3 ring =
            float3(direction == waveDirectionX ? 0.0f : p.x, direction == waveDirectionY ? 0.0f : p.y, direction == waveDirectionZ ? 0.0f : p.z);
        n = length(ring) * 20.0f;
    }
    n += shape.y;
    if (distortion != 0.0f)
        n += distortion * (fractalNoise(p * settings.w, settings.z, shape.x) * 2.0f - 1.0f);
    const uint profile = mode >> 3u;
    if (profile == waveProfileSaw) {
        const float t = n / (2.0f * programPi);
        return t - floor(t);
    }
    if (profile == waveProfileTriangle) {
        const float t = n / (2.0f * programPi);
        return abs(t - floor(t + 0.5f)) * 2.0f;
    }
    return 0.5f + 0.5f * sin(n - 0.5f * programPi);
}

NR_SHARED float safeDivide(float a, float b) {
    return b != 0.0f ? a / b : 0.0f;
}
NR_SHARED float safeModulo(float a, float b) {
    return b != 0.0f ? fmod(a, b) : 0.0f;
}
NR_SHARED float flooredModulo(float a, float b) {
    return b != 0.0f ? a - floor(a / b) * b : 0.0f;
}
NR_SHARED float safePower(float a, float b) {
    if (a < 0.0f && b != floor(b))
        return 0.0f;
    return pow(abs(a), b) * (a < 0.0f && fmod(abs(b), 2.0f) == 1.0f ? -1.0f : 1.0f);
}
NR_SHARED float safeLogarithm(float a, float b) {
    return a > 0.0f && b > 0.0f && b != 1.0f ? log(a) / log(b) : 0.0f;
}
NR_SHARED float signOf(float a) {
    return a > 0.0f ? 1.0f : a < 0.0f ? -1.0f : 0.0f;
}
NR_SHARED float truncate(float a) {
    return a >= 0.0f ? floor(a) : ceil(a);
}
NR_SHARED float wrapValue(float value, float high, float low) {
    const float range = high - low;
    return range != 0.0f ? value - range * floor((value - low) / range) : low;
}
NR_SHARED float snapValue(float a, float b) {
    return b != 0.0f ? floor(a / b) * b : 0.0f;
}
NR_SHARED float pingPong(float a, float b) {
    if (b == 0.0f)
        return 0.0f;
    const float t = (a - b) / (b * 2.0f);
    return abs((t - floor(t)) * b * 2.0f - b);
}
NR_SHARED float smoothMinimum(float a, float b, float c) {
    if (c == 0.0f)
        return min(a, b);
    const float h = max(c - abs(a - b), 0.0f) / c;
    return min(a, b) - h * h * h * c * (1.0f / 6.0f);
}

NR_SHARED float evaluateMath(uint operation, float a, float b, float c) {
    switch (operation & 127u) {
    case mathAdd:
        return a + b;
    case mathSubtract:
        return a - b;
    case mathMultiply:
        return a * b;
    case mathDivide:
        return safeDivide(a, b);
    case mathMultiplyAdd:
        return a * b + c;
    case mathPower:
        return safePower(a, b);
    case mathLogarithm:
        return safeLogarithm(a, b);
    case mathSqrt:
        return a > 0.0f ? sqrt(a) : 0.0f;
    case mathInverseSqrt:
        return a > 0.0f ? 1.0f / sqrt(a) : 0.0f;
    case mathAbsolute:
        return abs(a);
    case mathExponent:
        return exp(a);
    case mathMinimum:
        return min(a, b);
    case mathMaximum:
        return max(a, b);
    case mathLessThan:
        return a < b ? 1.0f : 0.0f;
    case mathGreaterThan:
        return a > b ? 1.0f : 0.0f;
    case mathSign:
        return signOf(a);
    case mathCompare:
        return abs(a - b) <= max(c, 0.00001f) ? 1.0f : 0.0f;
    case mathSmoothMin:
        return smoothMinimum(a, b, c);
    case mathSmoothMax:
        return -smoothMinimum(-a, -b, c);
    case mathRound:
        return floor(a + 0.5f);
    case mathFloor:
        return floor(a);
    case mathCeil:
        return ceil(a);
    case mathTruncate:
        return truncate(a);
    case mathFraction:
        return a - floor(a);
    case mathTruncatedModulo:
        return safeModulo(a, b);
    case mathFlooredModulo:
        return flooredModulo(a, b);
    case mathWrap:
        return wrapValue(a, b, c);
    case mathSnap:
        return snapValue(a, b);
    case mathPingPong:
        return pingPong(a, b);
    case mathSine:
        return sin(a);
    case mathCosine:
        return cos(a);
    case mathTangent:
        return tan(a);
    case mathArcsine:
        return asin(clamp(a, -1.0f, 1.0f));
    case mathArccosine:
        return acos(clamp(a, -1.0f, 1.0f));
    case mathArctangent:
        return atan(a);
    case mathArctan2:
        return atan2(a, b);
    case mathSinh:
        return sinh(a);
    case mathCosh:
        return cosh(a);
    case mathTanh:
        return tanh(a);
    case mathRadians:
        return a * (programPi / 180.0f);
    case mathDegrees:
        return a * (180.0f / programPi);
    default:
        return 0.0f;
    }
}

NR_SHARED float3 safeNormalize(float3 v) {
    const float size = length(v);
    return size > 0.0f ? v / size : float3(0.0f, 0.0f, 0.0f);
}

NR_SHARED float3 refractVector(float3 incident, float3 normal, float ior) {
    const float cosine = dot(normal, incident);
    const float k = 1.0f - ior * ior * (1.0f - cosine * cosine);
    return k < 0.0f ? float3(0.0f, 0.0f, 0.0f) : incident * ior - normal * (ior * cosine + sqrt(k));
}

NR_SHARED float4 evaluateVectorMath(uint operation, float3 a, float3 b, float3 c, float scale) {
    switch (operation) {
    case vectorMathAdd:
        return fromVector(a + b);
    case vectorMathSubtract:
        return fromVector(a - b);
    case vectorMathMultiply:
        return fromVector(a * b);
    case vectorMathDivide:
        return float4(safeDivide(a.x, b.x), safeDivide(a.y, b.y), safeDivide(a.z, b.z), 0.0f);
    case vectorMathMultiplyAdd:
        return fromVector(a * b + c);
    case vectorMathCrossProduct:
        return fromVector(cross(a, b));
    case vectorMathProject: {
        const float size = dot(b, b);
        return size != 0.0f ? fromVector(b * (dot(a, b) / size)) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    case vectorMathReflect: {
        const float3 n = safeNormalize(b);
        return fromVector(a - n * (2.0f * dot(n, a)));
    }
    case vectorMathRefract:
        return fromVector(refractVector(a, safeNormalize(b), scale));
    case vectorMathFaceforward:
        return fromVector(dot(c, b) < 0.0f ? a : a * -1.0f);
    case vectorMathDotProduct:
        return broadcast(dot(a, b));
    case vectorMathDistance:
        return broadcast(length(a - b));
    case vectorMathLength:
        return broadcast(length(a));
    case vectorMathScale:
        return fromVector(a * scale);
    case vectorMathNormalize:
        return fromVector(safeNormalize(a));
    case vectorMathAbsolute:
        return float4(abs(a.x), abs(a.y), abs(a.z), 0.0f);
    case vectorMathPower:
        return float4(safePower(a.x, b.x), safePower(a.y, b.y), safePower(a.z, b.z), 0.0f);
    case vectorMathSign:
        return float4(signOf(a.x), signOf(a.y), signOf(a.z), 0.0f);
    case vectorMathMinimum:
        return float4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), 0.0f);
    case vectorMathMaximum:
        return float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), 0.0f);
    case vectorMathFloor:
        return float4(floor(a.x), floor(a.y), floor(a.z), 0.0f);
    case vectorMathCeil:
        return float4(ceil(a.x), ceil(a.y), ceil(a.z), 0.0f);
    case vectorMathFraction:
        return float4(a.x - floor(a.x), a.y - floor(a.y), a.z - floor(a.z), 0.0f);
    case vectorMathModulo:
        return float4(safeModulo(a.x, b.x), safeModulo(a.y, b.y), safeModulo(a.z, b.z), 0.0f);
    case vectorMathWrap:
        return float4(wrapValue(a.x, b.x, c.x), wrapValue(a.y, b.y, c.y), wrapValue(a.z, b.z, c.z), 0.0f);
    case vectorMathSnap:
        return float4(snapValue(a.x, b.x), snapValue(a.y, b.y), snapValue(a.z, b.z), 0.0f);
    case vectorMathSine:
        return float4(sin(a.x), sin(a.y), sin(a.z), 0.0f);
    case vectorMathCosine:
        return float4(cos(a.x), cos(a.y), cos(a.z), 0.0f);
    case vectorMathTangent:
        return float4(tan(a.x), tan(a.y), tan(a.z), 0.0f);
    default:
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

NR_SHARED float getComponent(float4 value, uint component) {
    return component == 0u ? value.x : component == 1u ? value.y : component == 2u ? value.z : value.w;
}

//! The inputs of a program at one vertex, in object space.
struct ProgramVertex {
    float3 position;
    float3 normal;
    float3 generated;
};

//! Evaluates every op but Attribute, which reads the vertex attributes of the caller.
NR_SHARED float4 evaluateProgramInstruction(uint2 instruction, float4 a, float4 b, float4 c, float4 d, ProgramVertex vertex) {
    const uint mode = getProgramMode(instruction);
    switch (getProgramOp(instruction)) {
    case programOpMath: {
        const float value = evaluateMath(mode, a.x, b.x, c.x);
        return broadcast(mode >= mathClamp ? saturate(value) : value);
    }
    case programOpVectorMath:
        return evaluateVectorMath(mode, toVector(a), toVector(b), toVector(c), d.x);
    case programOpWave:
        return broadcast(waveTexture(mode, toVector(a), b, c));
    case programOpCombine:
        return float4(a.x, b.x, c.x, d.x);
    case programOpSeparate:
        return broadcast(getComponent(a, mode));
    case programOpAverage:
        return broadcast((a.x + a.y + a.z) / 3.0f);
    case programOpGrayscale:
        return broadcast(0.2126f * a.x + 0.7152f * a.y + 0.0722f * a.z);
    case programOpPosition:
        return fromVector(vertex.position);
    case programOpNormal:
        return fromVector(vertex.normal);
    case programOpGenerated:
        return fromVector(vertex.generated);
    case programOpDisplacement:
        return fromVector(toVector(d) * ((a.x - b.x) * c.x));
    case programOpVectorDisplacement:
        return float4((a.x - b.x) * c.x, (a.y - b.x) * c.x, (a.z - b.x) * c.x, 0.0f);
    default:
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

NR_NAMESPACE_END
