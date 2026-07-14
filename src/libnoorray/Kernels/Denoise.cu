#include "Raytracing/SceneData.h"

NR_GPU_KERNEL void writeDenoisedOutputKernel(
    const KernelParams params, const glm::vec4* denoised)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t pixelCount = params.frame.width * params.frame.height;
    if (pixel >= pixelCount)
        return;

    const glm::vec4 value = denoised[pixel];
    const uint32_t x = pixel % params.frame.width;
    const uint32_t y = pixel / params.frame.width;
    surf2Dwrite(make_float4(value.x, value.y, value.z, value.w),
        params.output.color, x * sizeof(float4), y);
}
