#ifndef _COMMON_SLANG_ //include guard to prevent multiple inclusions
#define _COMMON_SLANG_


static const float PI = 3.14159265358979323846;
static const float EPSILON = 0.0001;
static const float AABB_EPSILON  = 1e-5;
static const float INF = 1.0e30;
static const float FIREFLY_ABSOLUTE_LUMINANCE = 32.0;
static const float FIREFLY_RELATIVE_LUMINANCE = 12.0;
static const float FIREFLY_MIN_REFERENCE = 0.25;

// Bounce Types
enum BounceType : uint
{
    RAY_TERMINATED    = 1u << 0, // bit 0
    RAY_TRANSPARENT   = 1u << 1, // bit 1 (alpha-tested geometry skip)
    BOUNCE_DIFFUSE    = 1u << 2, // bit 2
    BOUNCE_SPECULAR   = 1u << 3, // bit 3
    BOUNCE_TRANSMIT   = 1u << 4, // bit 4
    ENV_TRANSPARENT   = 1u << 5, // bit 5 (invisible environment background)
};

float luminance(float3 c) { 
    return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

float3 compressLuminancePreserveHue(float3 color, float maxLuminance) {
    float currentLuminance = luminance(color);
    if (currentLuminance <= maxLuminance)
        return color;

    float excess = currentLuminance - maxLuminance;
    float compressedLuminance = maxLuminance + excess / (1.0 + excess / max(maxLuminance, EPSILON));
    return color * (compressedLuminance / max(currentLuminance, EPSILON));
}

float3 suppressFireflies(
    float3 sampleRadiance,
    float3 referenceRadiance,
    int bounceCount,
    int diffuseCount,
    int specularCount,
    int transmissionCount)
{
    float sampleLuminance = luminance(sampleRadiance);
    if (sampleLuminance <= 0.0)
        return max(sampleRadiance, float3(0.0));

    float referenceLuminance = max(luminance(max(referenceRadiance, float3(0.0))), FIREFLY_MIN_REFERENCE);
    float allowedLuminance = max(FIREFLY_ABSOLUTE_LUMINANCE, referenceLuminance * FIREFLY_RELATIVE_LUMINANCE);

    float glossyRisk = float(specularCount + transmissionCount);
    float deepRisk = max(float(bounceCount) - 1.0, 0.0);
    float diffuseRelief = 1.0 / (1.0 + 0.2 * float(diffuseCount));
    float riskScale = (1.0 + glossyRisk + 0.35 * deepRisk) * diffuseRelief;
    allowedLuminance /= max(riskScale, 1.0);

    return compressLuminancePreserveHue(max(sampleRadiance, float3(0.0)), allowedLuminance);
}

float powerHeuristic(float pdfA, float pdfB) {
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / max(a2 + b2, EPSILON);
}

float computeRussianRouletteSurvivalProbability(float3 throughput, int bounce, int startBounce)
{
    if (bounce < startBounce)
        return 1.0;

    float pathEnergy = max(throughput.r, max(throughput.g, throughput.b));
    float depthPenalty = 1.0 / (1.0 + 0.12 * float(max(bounce - startBounce, 0)));
    return clamp(pathEnergy * depthPenalty, 0.1, 0.95);
}

// Barycentric Helpers
float3 calculateBarycentric(float3 attribs) {
    return float3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
}

float3 interpolateBarycentric(float3 bary, float3 p0, float3 p1, float3 p2) {
    return p0 * bary.x + p1 * bary.y + p2 * bary.z;
}

float2 interpolateBarycentric(float3 bary, float2 p0, float2 p1, float2 p2) {
    return p0 * bary.x + p1 * bary.y + p2 * bary.z;
}

// PCG Random Number Generation
uint pcg(inout uint state)
{
    uint prev = state * 747796405u + 2891336453u;
    uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    state = prev;
    return (word >> 22u) ^ word;
}

uint2 pcg2d(uint2 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    return v;
}

// RNG float in [0,1)
float rand(inout uint seed)
{
    uint val = pcg(seed);
    return float(val) * (1.0 / 4294967296.0);
}

float computeShadowBias(float3 worldPosition, float nDotLGeom) {
    float sceneScale = max(max(abs(worldPosition.x), abs(worldPosition.y)), abs(worldPosition.z));
    float baseBias = max(2e-4, sceneScale * 2e-6);
    float grazingScale = 1.0 / max(nDotLGeom, 0.1);
    return baseBias * grazingScale;
}

float computeRayOriginBias(float3 worldPosition) {
    float sceneScale = max(max(abs(worldPosition.x), abs(worldPosition.y)), abs(worldPosition.z));
    return max(1e-4, sceneScale * 1e-6);
}

float3 offsetRayOrigin(float3 worldPosition, float3 geometricNormal, float3 outDirection) {
    float bias = computeRayOriginBias(worldPosition);
    float side = (dot(outDirection, geometricNormal) >= 0.0) ? 1.0 : -1.0;
    return worldPosition + geometricNormal * (side * bias);
}

float3 offsetShadowRayOrigin(
    float3 worldPosition,
    float3 worldShadowPosition,
    float3 geometricNormal,
    float3 shadowDirection,
    float nDotLGeom,
    out float tMin)
{
    float bias = computeShadowBias(worldPosition, nDotLGeom);
    float side = (dot(shadowDirection, geometricNormal) >= 0.0) ? 1.0 : -1.0;
    tMin = max(computeRayOriginBias(worldPosition), bias * 0.25);
    return worldShadowPosition + geometricNormal * (side * bias);
}

float3 sampleMaterialShadingNormal(
    int normalTextureIndex,
    float2 interpolatedUV,
    float3 shadingNormal,
    float3 interpolatedTangent,
    float interpolatedTangentSign)
{
    if (normalTextureIndex == -1)
        return shadingNormal;

    float3 packedNormal = textureSamplers[normalTextureIndex].Sample(interpolatedUV).xyz;
    return applyNormalMap(shadingNormal, interpolatedTangent, interpolatedTangentSign, packedNormal);
}

float3 computeFacingGeometricNormal(float3 geometricNormal, float3 viewDir)
{
    return dot(geometricNormal, viewDir) < 0.0 ? -geometricNormal : geometricNormal;
}

float3 computeFacingShadingNormal(float3 shadingNormal, float3 viewDir, float3 geometricFacingNormal)
{
    float3 shadingFacingNormal = dot(shadingNormal, viewDir) < 0.0 ? -shadingNormal : shadingNormal;
    shadingFacingNormal = stabilizeShadingNormal(shadingFacingNormal, geometricFacingNormal);
    return adaptShadingNormalLocal(viewDir, shadingFacingNormal, geometricFacingNormal);
}

#endif
