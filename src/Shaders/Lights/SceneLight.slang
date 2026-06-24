#ifndef SCENE_LIGHT_SLANG
#define SCENE_LIGHT_SLANG

#include "../Bindings.slang"
#include "../Core/Core.slang"
#include "ILight.slang"

// Light types matching LightType enum in C#
static const int LIGHT_DIRECTIONAL = 0;
static const int LIGHT_POINT = 1;
static const int LIGHT_SPOT = 2;
static const int LIGHT_RECT = 3;
static const int LIGHT_SKY = 4;

bool isSceneLightActive(int lightIndex)
{
    LightGpu light = sceneLightsBuffer[lightIndex];
    return light.intensity > EPSILON;
}

float computePointLightAttenuation(float distance, float range, float falloffExponent, bool useInverseSquared)
{
    if (distance >= range)
        return 0.0;

    float distanceRatio = distance / range;
    float attenuation;

    if (useInverseSquared) {
        float invDistSq = 1.0 / max(distance * distance, EPSILON);
        float rangeAtten = 1.0 - (distanceRatio * distanceRatio * distanceRatio * distanceRatio);
        attenuation = invDistSq * rangeAtten * rangeAtten;
    } else {
        float distAtten = max(1.0 - distanceRatio, 0.0);
        attenuation = pow(distAtten, falloffExponent);
    }

    return attenuation;
}

float computeSpotLightAttenuation(float3 lightToSurface, float3 lightDirection, float innerConeCos, float outerConeCos)
{
    float cosAngle = dot(normalize(lightToSurface), normalize(lightDirection));

    if (cosAngle < outerConeCos)
        return 0.0;

    if (innerConeCos >= outerConeCos)
        return 1.0;

    return smoothstep(outerConeCos, innerConeCos, cosAngle);
}

LightLiSample sampleSceneLight(int lightIndex, float3 referencePoint, float2 u)
{
    LightGpu light = sceneLightsBuffer[lightIndex];
    LightLiSample sample;
    sample.pdf = 0.0;
    sample.Li = float3(0.0);
    sample.wi = float3(0.0);

    if (!isSceneLightActive(lightIndex))
        return sample;

    if (light.type == LIGHT_DIRECTIONAL) {
        float3 dir = normalize(light.direction);
        float softAngle = light.lightSourceSoftAngle;
        if (softAngle > EPSILON) {
            float cosThetaMax = cos(radians(softAngle));
            float cosTheta = 1.0 - u.x * (1.0 - cosThetaMax);
            float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
            float phi = 2.0 * PI * u.y;
            float3 tangent, bitangent;
            float s = dir.z >= 0.0 ? 1.0 : -1.0;
            float a = -1.0 / (s + dir.z);
            float b = dir.x * dir.y * a;
            tangent = float3(1.0 + s * dir.x * dir.x * a, s * b, -s * dir.x);
            bitangent = float3(b, s + dir.y * dir.y * a, -dir.y);
            dir = normalize(tangent * (cos(phi) * sinTheta) + bitangent * (sin(phi) * sinTheta) + dir * cosTheta);
        }

        sample.wi = dir;
        sample.Li = float3(light.intensity);
        sample.pdf = 1.0;
        return sample;
    }

    float3 lightPos = light.position;
    float3 toLight = lightPos - referencePoint;
    float dist = length(toLight);
    float3 wi = toLight / max(dist, EPSILON);

    if (light.type == LIGHT_POINT || light.type == LIGHT_SPOT) {
        float attenuation = computePointLightAttenuation(dist, light.range, light.lightFalloffExponent, light.useInverseSquaredFalloff != 0);
        if (attenuation <= EPSILON)
            return sample;

        if (light.type == LIGHT_SPOT) {
            float spotAtten = computeSpotLightAttenuation(toLight, light.direction,
                cos(radians(light.innerConeAngle)),
                cos(radians(light.outerConeAngle)));
            attenuation *= spotAtten;
            if (attenuation <= EPSILON)
                return sample;
        }

        sample.wi = wi;
        sample.Li = light.color * light.intensity * attenuation;
        sample.pdf = 1.0;
        return sample;
    }

    if (light.type == LIGHT_RECT) {
        float area = max(light.sourceWidth * light.sourceHeight * 0.0001f, EPSILON);
        float3 rectDir = normalize(light.direction);
        float attenuation = computePointLightAttenuation(dist, light.range, light.lightFalloffExponent, light.useInverseSquaredFalloff != 0);
        if (attenuation <= EPSILON)
            return sample;

        sample.wi = wi;
        sample.Li = light.color * light.intensity * attenuation / max(PI, EPSILON);
        sample.pdf = 1.0;
        return sample;
    }

    return sample;
}

BxdfEval evalSceneLightBSDF(
    int lightIndex,
    float3 viewDir,
    float3 shadingNormal,
    float3 geometricNormal,
    float3 worldPosition,
    float3 albedo,
    float specular,
    float metallic,
    float roughness,
    float swSpecular,
    inout uint rngState)
{
    LightGpu light = sceneLightsBuffer[lightIndex];
    PbrBxdf bxdf;
    float2 u = float2(rand(rngState), rand(rngState));
    LightLiSample lightSample = sampleSceneLight(lightIndex, worldPosition, u);

    BxdfEval bxdfEval;
    bxdfEval.f = float3(0.0);
    bxdfEval.pdf = 0.0;

    if (lightSample.pdf <= EPSILON || dot(lightSample.Li, lightSample.Li) <= EPSILON)
        return bxdfEval;

    float NdotLGeom = max(dot(geometricNormal, lightSample.wi), 0.0);
    if (NdotLGeom <= 0.0)
        return bxdfEval;

    bxdfEval = bxdf.evalOpaque(
        viewDir, shadingNormal, geometricNormal,
        lightSample.wi, albedo, specular, metallic, roughness, swSpecular);

    if (bxdfEval.pdf <= 0.0) {
        bxdfEval.f = float3(0.0);
        return bxdfEval;
    }

    bxdfEval.f *= lightSample.Li * NdotLGeom / max(lightSample.pdf, EPSILON);
    return bxdfEval;
}

#endif // SCENE_LIGHT_SLANG
