import NoorRHI;

static const float pi = 3.14159265358979;
static const float halfPi = 1.57079632679490;
static const float twoPi = 6.28318530717959;
static const float fourPi = 12.5663706143592;
static const float invPi = 0.318309886183791;
static const float invTwoPi = 0.159154943091895;
static const float invFourPi = 0.0795774715459477;
static const float floatMax = 3.402823e38;
static const float oneMinusEpsilon = 0.99999994;

float square(float x) { return x * x; }
float3 square(float3 x) { return x * x; }
float maxComponent(float3 v) { return max(v.x, max(v.y, v.z)); }
float luminance(float3 rgb) { return dot(rgb, float3(0.2126, 0.7152, 0.0722)); }
float safeSqrt(float x) { return sqrt(max(x, 0.0)); }
float safeDivide(float a, float b) { return b != 0.0 ? a / b : 0.0; }
float3 safeNormalize(float3 v) {
    const float lengthSquared = dot(v, v);
    return lengthSquared > 0.0 ? v * rsqrt(lengthSquared) : float3(0.0, 0.0, 1.0);
}
bool isFiniteValue(float3 v) { return all(!isnan(v) && !isinf(v)); }
float3 sanitize(float3 v) { return isFiniteValue(v) ? v : float3(0.0, 0.0, 0.0); }

//! Rows of a 3x4 affine transform, as InstanceData stores them.
float3 transformPoint(float4 rows[3], float3 p) { return float3(dot(rows[0].xyz, p) + rows[0].w, dot(rows[1].xyz, p) + rows[1].w, dot(rows[2].xyz, p) + rows[2].w); }
float3 transformVector(float4 rows[3], float3 v) { return float3(dot(rows[0].xyz, v), dot(rows[1].xyz, v), dot(rows[2].xyz, v)); }
//! A normal goes through the transpose of the inverse; rows here are those of the inverse (worldToObject).
float3 transformNormal(float4 inverseRows[3], float3 n) {
    return inverseRows[0].xyz * n.x + inverseRows[1].xyz * n.y + inverseRows[2].xyz * n.z;
}

float3 unpackOctahedral(uint packed) {
    const float u = max(float(int(packed << 16) >> 16) / 32767.0, -1.0);
    const float v = max(float(int(packed) >> 16) / 32767.0, -1.0);
    float3 n = float3(u, v, 1.0 - abs(u) - abs(v));
    const float t = max(-n.z, 0.0);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

uint packOctahedral(float3 direction) {
    const float3 n = direction / (abs(direction.x) + abs(direction.y) + abs(direction.z));
    float2 uv = n.xy;
    if (n.z < 0.0)
        uv = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    const int2 snorm = int2(round(clamp(uv, -1.0, 1.0) * 32767.0));
    return (uint(snorm.x) & 0xffffu) | (uint(snorm.y) << 16);
}

//! The tangent keeps 15 bits of the second octahedral coordinate; bit 31 is the bitangent sign.
float4 unpackTangent(uint packed) {
    const int v15 = int(packed << 1) >> 17;
    const uint octahedral = (packed & 0xffffu) | (uint(v15 * 2) << 16);
    return float4(unpackOctahedral(octahedral), (packed & 0x80000000u) != 0 ? -1.0 : 1.0);
}

float2 unpackHalf2(uint packed) { return float2(f16tof32(packed & 0xffffu), f16tof32(packed >> 16)); }
uint packHalf2(float2 value) { return f32tof16(value.x) | (f32tof16(value.y) << 16); }

float4 unpackUnorm4x8(uint packed) {
    return float4(float(packed & 0xffu), float((packed >> 8) & 0xffu), float((packed >> 16) & 0xffu), float(packed >> 24)) / 255.0;
}

uint packUnorm4x8(float4 value) {
    const uint4 bytes = uint4(round(saturate(value) * 255.0));
    return bytes.x | (bytes.y << 8) | (bytes.z << 16) | (bytes.w << 24);
}

float3 rotateQuaternion(float4 q, float3 v) {
    const float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float3 rotateY(float3 v, float angle) {
    const float c = cos(angle);
    const float s = sin(angle);
    return float3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

//! The equirect mapping of Scene/Environment.h: u = 0.5 looks down -Z, v = 0 is +Y.
float2 directionToEquirect(float3 direction) {
    return float2(atan2(direction.x, -direction.z) * invTwoPi + 0.5, acos(clamp(direction.y, -1.0, 1.0)) * invPi);
}

float3 equirectToDirection(float2 uv) {
    const float phi = (uv.x - 0.5) * twoPi;
    const float theta = uv.y * pi;
    return float3(sin(theta) * sin(phi), cos(theta), -sin(theta) * cos(phi));
}
