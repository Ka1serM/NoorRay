#include <cuda_fp16.h>

#include "Raytracing/SceneData.h"

NR_GPU_KERNEL void finalizeKernel(const KernelParams params)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t pixelCount = params.frame.width * params.frame.height;
    if (pixel >= pixelCount)
        return;

    const PathState state = params.queues.pathStates[pixel];
    const float alpha = static_cast<float>((state.packedCounters >> CounterHitShift) & 1u);

    // Spectral radiance → linear sRGB via CIE XYZ.
    const glm::vec3 xyz = spectrumToXYZ(
        state.radiance, state.wl, params.scene.cieX, params.scene.cieY, params.scene.cieZ);
    glm::vec3 radiance = xyzToLinearSRGB(xyz);

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
}
