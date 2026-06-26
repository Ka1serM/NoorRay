#ifndef RAYTRACING_GENERATE_RAY_LUT_SLANG
#define RAYTRACING_GENERATE_RAY_LUT_SLANG

#include "../Shared.h"

struct RayLutPushData
{
    int width;
    int height;
    int _pad0;
    int _pad1;
};

[[vk::push_constant]] ConstantBuffer<RayLutPushData> pushConstants;
[[vk::binding(0, 0)]] StructuredBuffer<SceneSettings, ScalarDataLayout> sceneSettingsBuffer;
#define sceneSettings sceneSettingsBuffer[0]
[[vk::binding(1, 0)]] RWStructuredBuffer<RayLutEntry, Std430DataLayout> cameraRayLutWrite;

#define RAY_LUT_GENERATION_SHADER 1
#include "../Cameras/RealisticCamera.slang"

[shader("compute")]
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 pixel = int2(dispatchThreadID.xy);
    int2 screenSize = int2(pushConstants.width, pushConstants.height);
    if (pixel.x >= screenSize.x || pixel.y >= screenSize.y)
        return;

    uint lutIndex = uint(pixel.x + pixel.y * screenSize.x);
    RayLutEntry entry;
    entry.origin = float3(0.0);
    entry.originValid = 0.0;
    entry.direction = float3(0.0, 0.0, 1.0);
    entry._pad0 = 0.0;

    if (sceneSettings.camera.cameraType == CAMERA_REALISTIC) {
        LocalRay localRay;
        if (generateRealisticLocalRay(pixel, screenSize, float2(0.0), float2(0.5), localRay)) {
            entry.origin = localRay.origin;
            entry.originValid = 1.0;
            entry.direction = normalize(localRay.direction);
        }
    }

    cameraRayLutWrite[lutIndex] = entry;
}

#endif // RAYTRACING_GENERATE_RAY_LUT_SLANG
