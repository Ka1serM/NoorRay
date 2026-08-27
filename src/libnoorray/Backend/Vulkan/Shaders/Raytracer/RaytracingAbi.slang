#pragma once

// Shared ABI for all ray-tracing stages. There are no layout(binding)
// declarations here: resources are formed from heap indices in FrameParams,
// and the only root value sent by the host is the argument pointer.
struct FrameParams
{
    uint64_t sceneAddress;
    uint64_t cameraAddress;
    uint64_t lensAddress;
    uint64_t materialsAddress;
    uint colorImage;
    uint albedoImage;
    uint normalImage;
    uint positionImage;
    uint cryptomatteImage;
    uint lensBuffer;
    uint svmWords;
    uint topLevelAS;
    uint width;
    uint height;
    uint frameIndex;
    uint sampleIndex;
    float exposure;
    float shutterOpen;
    float shutterClose;
    uint sceneBuffer;
    uint gaussianRecords;
    uint gaussianOpacities;
    uint gaussianShCoefficients;
    uint gaussianInstanceOffsets;
    uint gaussianCount;
    uint gaussianShCoefficientCount;
    float gaussianCutoffDistanceSq;
    uint accumulationBuffer;
    uint environmentBuffer;
    uint energyLutBuffer;
    uint spectralTables;
    uint maxBounces;
    float indirectLightClamp;
    uint transparentBackground;
    uint gaussianProxyTriangleCount;
    uint gaussianOverdrawImage;
    uint gaussianOverdrawEnabled;
    uint gaussianOverdrawMax;
    uint aovEnabled;
    uint gaussianInstanceBase;
};

struct RaytracerRoot
{
    FrameParams* arguments;
};

[[vk::push_constant]] ConstantBuffer<RaytracerRoot> root;

struct RayPayload
{
    uint hit;
    uint instance;
    uint primitive;
    float2 barycentrics;
    float t;
    uint seed;
    uint gaussianId;
    float gaussianOpacity;
    float3 radiance;
    float3 albedo;
    float3 normal;
    float3 position;
    float alpha;
    uint cryptomatte;
    float4 bsdfWeight;
    float4 emissionSpectrum;
    float3 nextDirection;
    uint sampleValid;
    float4 directSpectrum;
    float3 shadowDirection;
    float shadowDistance;
    uint shadowValid;
    uint shadowInstance;
    uint shadowPrimitive;
    float wavelengthSample;
    uint depth;
    float bsdfPdf;
    float lightPdf;
};
