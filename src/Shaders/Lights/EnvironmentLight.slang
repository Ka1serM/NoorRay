#ifndef ENVIRONMENT_LIGHT_SLANG
#define ENVIRONMENT_LIGHT_SLANG

#include "../Bindings.slang"
#include "../Core/Core.slang"
#include "../Bxdfs/Brdf.slang"
#include "ILight.slang"

float2 directionToEnvironmentUv(in float3 worldRayDirection, in EnvironmentSettings environmentData) {
    float3 rotatedDir = worldRayDirection;
    rotatedDir.x = worldRayDirection.x * environmentData.rotationCos - worldRayDirection.z * environmentData.rotationSin;
    rotatedDir.z = worldRayDirection.x * environmentData.rotationSin + worldRayDirection.z * environmentData.rotationCos;

    float2 uv;
    uv.x = atan2(rotatedDir.z, rotatedDir.x) / (2.0 * PI) + 0.5;
    uv.y = acos(clamp(rotatedDir.y, -1.0, 1.0)) / PI;
    uv.x = frac(uv.x);
    uv.y = saturate(uv.y);
    return uv;
}

float3 sampleEnvironmentMapColor(in float3 worldRayDirection, in EnvironmentSettings environmentData, in float exposureStops) {
    if (environmentData.textureIndex == -1)
        return float3(1.0);

    float2 uv = directionToEnvironmentUv(worldRayDirection, environmentData);
    return textureSamplers[environmentData.textureIndex].SampleLevel(uv, 0.0).rgb * exposureStops;
}

float3 sampleEnvironmentMapColorLod(in float3 worldRayDirection, in EnvironmentSettings environmentData, in float exposureStops, in float lod) {
    if (environmentData.textureIndex == -1)
        return float3(1.0);

    float2 uv = directionToEnvironmentUv(worldRayDirection, environmentData);
    return textureSamplers[environmentData.textureIndex].SampleLevel(uv, max(lod, 0.0)).rgb * exposureStops;
}

float evaluateEnvironmentPdfForDirection(in float3 worldRayDirection, in EnvironmentSettings environmentData) {
    if (environmentData.textureIndex < 0 || environmentData.cdfTextureIndex < 0)
        return 0.0;

    uint envWidth;
    uint envHeight;
    textureSamplers[environmentData.textureIndex].GetDimensions(envWidth, envHeight);
    int2 envSize = int2(envWidth, envHeight);
    if (envSize.x <= 0 || envSize.y <= 0)
        return 0.0;

    float2 uv = directionToEnvironmentUv(worldRayDirection, environmentData);
    int x = clamp(int(floor(uv.x * float(envSize.x))), 0, envSize.x - 1);
    int y = clamp(int(floor(uv.y * float(envSize.y))), 0, envSize.y - 1);
    return max(textureSamplers[environmentData.cdfTextureIndex].Load(int3(int2(x, y), 0)).a, 0.0);
}

float evaluateNeeLightPdfForDirection(in float3 worldRayDirection, in EnvironmentSettings environmentData) {
    return evaluateEnvironmentPdfForDirection(worldRayDirection, environmentData);
}

float3 evaluateEnvironmentRadiance(in float3 worldRayDirection, in EnvironmentSettings environmentData) {
    if (environmentData.textureIndex == -1)
        return float3(1.0);

    float2 uv = directionToEnvironmentUv(worldRayDirection, environmentData);
    return textureSamplers[environmentData.textureIndex].SampleLevel(uv, 0.0).rgb * environmentData.lightingExposureScale;
}

float3 evaluateEnvironmentVisibleRadiance(in float3 worldRayDirection, in EnvironmentSettings environmentData) {
    if (environmentData.textureIndex == -1)
        return float3(1.0);

    float2 uv = directionToEnvironmentUv(worldRayDirection, environmentData);
    float visibleLod = min(max(environmentData.maxTextureLod, 0.0), 2.0);
    return textureSamplers[environmentData.textureIndex].SampleLevel(uv, visibleLod).rgb * environmentData.visibleExposureScale;
}

int binarySearchMarginalCdf(int cdfTextureIndex, int height, float xi)
{
    int lo = 0;
    int hi = height - 1;
    for (int i = 0; i < 20 && lo < hi; i++) {
        int mid = (lo + hi) >> 1;
        float c = textureSamplers[cdfTextureIndex].Load(int3(int2(0, mid), 0)).g;
        if (c < xi)
            lo = mid + 1;
        else
            hi = mid;
    }
    return clamp(lo, 0, height - 1);
}

int binarySearchConditionalCdf(int cdfTextureIndex, int width, int y, float xi)
{
    int lo = 0;
    int hi = width - 1;
    for (int i = 0; i < 20 && lo < hi; i++) {
        int mid = (lo + hi) >> 1;
        float c = textureSamplers[cdfTextureIndex].Load(int3(int2(mid, y), 0)).r;
        if (c < xi)
            lo = mid + 1;
        else
            hi = mid;
    }
    return clamp(lo, 0, width - 1);
}

float3 sampleEnvironmentDirection(
    in EnvironmentSettings environmentData,
    inout uint rngState,
    out float lightPdf)
{
    lightPdf = 0.0;
    if (environmentData.textureIndex < 0 || environmentData.cdfTextureIndex < 0)
        return float3(0.0);

    uint envWidth;
    uint envHeight;
    textureSamplers[environmentData.textureIndex].GetDimensions(envWidth, envHeight);
    int2 envSize = int2(envWidth, envHeight);
    if (envSize.x <= 0 || envSize.y <= 0)
        return float3(0.0);

    float xi1 = rand(rngState);
    float xi2 = rand(rngState);

    int y = binarySearchMarginalCdf(environmentData.cdfTextureIndex, envSize.y, xi1);
    int x = binarySearchConditionalCdf(environmentData.cdfTextureIndex, envSize.x, y, xi2);
    float4 cdfTexel = textureSamplers[environmentData.cdfTextureIndex].Load(int3(int2(x, y), 0));

    float v = (float(y) + 0.5) / float(envSize.y);
    float u = (float(x) + 0.5) / float(envSize.x);
    float theta = v * PI;
    float phi = (u - 0.5) * (2.0 * PI);
    float sinTheta = max(sin(theta), EPSILON);

    lightPdf = max(cdfTexel.a, 0.0);

    float3 rotatedDir = float3(
        cos(phi) * sinTheta,
        cos(theta),
        sin(phi) * sinTheta);

    float3 worldDir = rotatedDir;
    worldDir.x = rotatedDir.x * environmentData.rotationCos + rotatedDir.z * environmentData.rotationSin;
    worldDir.z = -rotatedDir.x * environmentData.rotationSin + rotatedDir.z * environmentData.rotationCos;
    return fastNormalize(worldDir);
}

struct EnvironmentLight : ILight
{
    LightLiSample sampleLi(float3 referencePoint, float2 u)
    {
        referencePoint = referencePoint;

        LightLiSample sample;
        sample.wi = float3(0.0, 1.0, 0.0);
        sample.Li = float3(0.0);
        sample.pdf = 0.0;

        EnvironmentSettings environmentData = sceneSettings.environment;
        if (environmentData.textureIndex < 0 || environmentData.cdfTextureIndex < 0)
            return sample;

        uint rngState = asuint(u.x * 4294967295.0) ^ (asuint(u.y * 4294967295.0) * 747796405u + 2891336453u);
        if (rngState == 0u)
            rngState = 1u;

        sample.wi = sampleEnvironmentDirection(environmentData, rngState, sample.pdf);
        sample.Li = sampleEnvironmentMapColor(sample.wi, environmentData, environmentData.lightingExposureScale);
        return sample;
    }

    float pdfLi(float3 referencePoint, float3 wi)
    {
        referencePoint = referencePoint;

        EnvironmentSettings environmentData = sceneSettings.environment;
        if (environmentData.textureIndex < 0 || environmentData.cdfTextureIndex < 0)
            return 0.0;

        return evaluateEnvironmentPdfForDirection(wi, environmentData);
    }

    LightLeSample sampleLe(float2 u1, float2 u2)
    {
        u2 = u2;

        LightLeSample sample;
        sample.pdfPos = 1.0;
        sample.direction = float3(0.0, 1.0, 0.0);
        sample.radiance = float3(0.0);
        sample.pdfDir = 0.0;

        LightLiSample li = sampleLi(float3(0.0), u1);
        sample.direction = li.wi;
        sample.radiance = li.Li;
        sample.pdfDir = li.pdf;
        return sample;
    }
};

#endif // ENVIRONMENT_LIGHT_SLANG
