#include <cuda_fp16.h>

#include "Camera/Camera.h"
#include "Raytracing/Gpu/SceneData.h"

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

NR_GPU_KERNEL void writeDenoiserGuidesKernel(
    const KernelParams params, float3* albedoGuide, float3* normalGuide)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t pixelCount = params.frame.width * params.frame.height;
    if (pixel >= pixelCount)
        return;

    const uint32_t x = pixel % params.frame.width;
    const uint32_t y = pixel / params.frame.width;
    const uchar4 packed = surf2Dread<uchar4>(
        params.output.albedo, x * sizeof(uchar4), y);
    constexpr float Unorm8 = 1.0f / 255.0f;
    albedoGuide[pixel] = make_float3(
        packed.x * Unorm8, packed.y * Unorm8, packed.z * Unorm8);

    const ushort4 packedNormal = surf2Dread<ushort4>(
        params.output.normal, x * sizeof(ushort4), y);
    normalGuide[pixel] = make_float3(
        __half2float(__ushort_as_half(packedNormal.x)),
        __half2float(__ushort_as_half(packedNormal.y)),
        __half2float(__ushort_as_half(packedNormal.z)));
}

NR_GPU_KERNEL void postProcessKernel(
    const KernelParams params, float* varianceSum, const uint32_t flags)
{
    constexpr uint32_t BlockSize = 256;
    __shared__ float partial[BlockSize];

    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t pixelCount = params.frame.width * params.frame.height;

    if (pixel < pixelCount && (flags & 1u) != 0)
    {
        const Sensor& sensor = params.scene.camera->Dispatch(
            [](const auto* camera) -> const Sensor& { return camera->getSensor(); });
        if (const ScatterPsfSensor* scatterSensor = sensor.CastOrNullptr<ScatterPsfSensor>())
        {
            const glm::vec4 out = scatterSensor->resolvePixel(
                pixel, params.frame.width, params.accumulation);
            const uint32_t x = pixel % params.frame.width;
            const uint32_t y = pixel / params.frame.width;
            surf2Dwrite(make_float4(out.x, out.y, out.z, out.w),
                params.output.color, x * sizeof(float4), y);
        }
        else if (const GatherPsfSensor* gatherSensor = sensor.CastOrNullptr<GatherPsfSensor>())
        {
            if (params.psfGatherBuckets != nullptr && params.psfBinCount != 0)
            {
                const int destX = static_cast<int>(pixel % params.frame.width);
                const int destY = static_cast<int>(pixel / params.frame.width);
                const glm::vec4 out = gatherSensor->resolvePixel(
                    pixel, params.frame.width, params.frame.height,
                    params.psfBinCount, params.psfGatherBuckets);
                params.accumulation[pixel] = out;
                surf2Dwrite(make_float4(out.x, out.y, out.z, out.w),
                    params.output.color, destX * sizeof(float4), destY);
            }
        }
    }

    if ((flags & 2u) != 0)
    {
        const uint32_t sampleCount = params.frame.totalAccumulated + 1;
        float varianceOfMean = 0.0f;
        if (pixel < pixelCount && sampleCount > 1 && params.noiseMoments != nullptr)
        {
            const float m2 = params.noiseMoments[pixel].y;
            varianceOfMean = fmaxf(m2, 0.0f)
                / (static_cast<float>(sampleCount - 1) * static_cast<float>(sampleCount));
        }

        partial[threadIdx.x] = varianceOfMean;
        __syncthreads();
        for (uint32_t stride = BlockSize / 2; stride != 0; stride /= 2)
        {
            if (threadIdx.x < stride)
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            atomicAdd(varianceSum, partial[0]);
    }
}
