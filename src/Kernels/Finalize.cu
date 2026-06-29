#include <cuda_fp16.h>

#include "Raytracing/KernelHelpers.h"

NR_GPU_KERNEL void finalizeKernel(const KernelParams params)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t pixelCount = params.frame.width * params.frame.height;
    if (pixel >= pixelCount || adaptiveSkip(params, pixel))
        return;

    const PathState state = params.queues.pathStates[pixel];
    glm::vec3 radiance = state.radiance;
    const float alpha = static_cast<float>((state.packedCounters >> CounterHitShift) & 1u);

    const uint32_t x = pixel % params.frame.width;
    const uint32_t y = pixel / params.frame.width;
    const float weight = params.frame.totalAccumulated == 0
        ? 1.0f : 1.0f / static_cast<float>(params.frame.totalAccumulated + 1);
    const glm::vec4 previous = params.accumulation[pixel];
    radiance = glm::mix(glm::vec3(previous.x, previous.y, previous.z), radiance, weight);
    const float accAlpha = previous.w * (1.0f - weight) + alpha * weight;
    params.accumulation[pixel] = glm::vec4(radiance, accAlpha);
    surf2Dwrite(make_float4(radiance.x, radiance.y, radiance.z, accAlpha),
        params.output.color, x * sizeof(float4), y);

    const PrimaryState primary = params.queues.primaryStates[pixel];
    const glm::vec3 albedo = glm::clamp(primary.primaryAlbedo, 0.0f, 1.0f);
    surf2Dwrite(make_uchar4(static_cast<unsigned char>(albedo.x * 255.0f + 0.5f),
                            static_cast<unsigned char>(albedo.y * 255.0f + 0.5f),
                            static_cast<unsigned char>(albedo.z * 255.0f + 0.5f), 255),
                params.output.albedo, x * sizeof(uchar4), y);
    surf2Dwrite(packHalf4(primary.primaryNormal, 0.0f), params.output.normal, x * sizeof(ushort4), y);
    surf2Dwrite(packHalf4(primary.primaryPosition,
        primary.primaryObjectIndex == InvalidIndex ? 0.0f : 1.0f),
        params.output.position, x * sizeof(ushort4), y);

    if (params.frame.sampleIndex == 0)
        surf2Dwrite(primary.primaryObjectIndex, params.output.cryptomatte, x * sizeof(uint32_t), y);

    const float luminance = radiance.x * 0.2126f + radiance.y * 0.7152f + radiance.z * 0.0722f;
    float mean = luminance;
    float m2 = 0.0f;
    float count = 1.0f;
    if (params.frame.totalAccumulated > 0)
    {
        const glm::vec4 old = params.adaptiveState[pixel];
        count = old.z + 1.0f;
        const float delta = luminance - old.x;
        mean = old.x + delta / count;
        m2 = old.y + delta * (luminance - mean);
    }
    params.adaptiveState[pixel] = glm::vec4(mean, m2, count, 0.0f);
}
