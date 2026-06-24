#ifndef RAYTRACING_GENERATE_SLANG
#define RAYTRACING_GENERATE_SLANG

#include "../Shared.h"
#include "../Bindings.slang"

[[vk::push_constant]] ConstantBuffer<PushData> pushConstants;

#include "../Common.slang"
#include "../Core/AdaptiveSampling.slang"
#include "../Integrators.slang"

[shader("compute")]
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 shadingPixel = int2(dispatchThreadID.xy);
    uint screenWidth, screenHeight;
    outputColor.GetDimensions(screenWidth, screenHeight);
    int2 screenSize       = int2(screenWidth, screenHeight);
    int2 shadingScreenSize = computeShadingScreenSize(screenSize);
    if (shadingPixel.x >= shadingScreenSize.x || shadingPixel.y >= shadingScreenSize.y)
        return;

    int2 blockStart   = shadingPixelStart(shadingPixel, screenSize, shadingScreenSize);
    int2 blockEnd     = shadingPixelEndExclusive(shadingPixel, screenSize, shadingScreenSize);
    AdaptiveSamplingGateContext adaptiveContext = loadAdaptiveSamplingGateContext(blockStart);
    if (shouldSkipAdaptiveSampling(adaptiveContext))
        return;

    int2 blockCenterPixel = clamp((blockStart + (blockEnd - int2(1))) / 2, int2(0), screenSize - 1);
    int  cameraType       = pushConstants.camera.cameraType;
    uint sampleCount      = integratorSampleCount();

    for (uint sampleIndex = 0u; sampleIndex < sampleCount; ++sampleIndex) {
        uint slot     = computeSampleSlot(shadingPixel, shadingScreenSize, sampleIndex, sampleCount);
        uint2 seed    = pcg2d(uint2(blockCenterPixel) ^ uint2((pushConstants.frame + int(sampleIndex)) * 16777619));
        uint rngState = seed.x ^ (sampleIndex * 747796405u + 2891336453u);

        R2Sampler    sampler;
        SamplerState samplerState;
        sampler.startPixel(blockCenterPixel, pushConstants.frame + int(sampleIndex), samplerState);

        bool   deterministicSample = pushConstants.frame == 0 && sampleIndex == 0u;
        float3 rayOrigin, rayDirection;

        if (cameraType == CAMERA_THINLENS) {
            ThinLensCamera<R2Sampler> camera;
            camera.generateRay(blockCenterPixel, screenSize, samplerState, sampler, deterministicSample, rayOrigin, rayDirection);
        } else if (cameraType == CAMERA_REALISTIC) {
            RealisticCamera<R2Sampler> camera;
            camera.generateRay(blockCenterPixel, screenSize, samplerState, sampler, deterministicSample, rayOrigin, rayDirection);
        } else if (cameraType == CAMERA_ORTHOGRAPHIC) {
            OrthographicCamera<R2Sampler> camera;
            camera.generateRay(blockCenterPixel, screenSize, samplerState, sampler, deterministicSample, rayOrigin, rayDirection);
        } else if (cameraType == CAMERA_FISHEYE) {
            FisheyeCamera<R2Sampler> camera;
            camera.generateRay(blockCenterPixel, screenSize, samplerState, sampler, deterministicSample, rayOrigin, rayDirection);
        } else {
            // CAMERA_PERSPECTIVE (default / fallback)
            PerspectiveCamera<R2Sampler> camera;
            camera.generateRay(blockCenterPixel, screenSize, samplerState, sampler, deterministicSample, rayOrigin, rayDirection);
        }

        initializePathState(slot, rayOrigin, rayDirection, rngState);

        PathRayWorkItem ray;
        ray.Origin      = rayOrigin;
        ray.SampleIndex = slot;
        ray.Direction   = rayDirection;
        ray.Pad0        = 0u;
        appendNextRay(ray);
    }
}

#endif // RAYTRACING_GENERATE_SLANG
