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

    SensorSampleContext ctx{};
    ctx.accumulation = params.accumulation;
    ctx.noiseMoments = params.noiseMoments;
    ctx.psfBuckets = params.psfGatherBuckets;
    ctx.width = params.frame.width;
    ctx.height = params.frame.height;
    ctx.totalAccumulated = params.frame.totalAccumulated;
    ctx.alpha = alpha;
    ctx.cieX = params.scene.cieX;
    ctx.cieY = params.scene.cieY;
    ctx.cieZ = params.scene.cieZ;

    const Sensor& sensor = params.scene.camera->Dispatch([](const auto* camera) -> const Sensor& {
        return camera->getSensor();
    });
    const bool deferredOutput = sensor.Is<ScatterPsfSensor>();
    sensor.Dispatch([&](const auto* concreteSensor) {
        concreteSensor->addSample(pixel, state.radiance, state.wl, 1.0f, ctx);
    });

    if (!deferredOutput)
    {
        const glm::vec4 out = params.accumulation[pixel];
        const uint32_t x = pixel % params.frame.width;
        const uint32_t y = pixel / params.frame.width;
        surf2Dwrite(make_float4(out.x, out.y, out.z, out.w),
            params.output.color, x * sizeof(float4), y);
    }
}

NR_GPU_KERNEL void reduceNoiseVarianceKernel(const KernelParams params, float* varianceSum)
{
    constexpr uint32_t BlockSize = 256;
    __shared__ float partial[BlockSize];

    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t pixelCount = params.frame.width * params.frame.height;
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

NR_GPU_KERNEL void resolveScatterPsfKernel(const KernelParams params)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t pixelCount = params.frame.width * params.frame.height;
    if (pixel >= pixelCount || params.accumulation == nullptr)
        return;

    const Sensor& sensor = params.scene.camera->Dispatch([](const auto* camera) -> const Sensor& {
        return camera->getSensor();
    });
    const ScatterPsfSensor* scatterSensor = sensor.CastOrNullptr<ScatterPsfSensor>();
    if (scatterSensor == nullptr)
        return;

    const glm::vec4 out = scatterSensor->resolvePixel(pixel, params.frame.width, params.accumulation);
    const uint32_t x = pixel % params.frame.width;
    const uint32_t y = pixel / params.frame.width;
    surf2Dwrite(make_float4(out.x, out.y, out.z, out.w),
        params.output.color, x * sizeof(float4), y);
}

NR_GPU_KERNEL void applyGatherPsfKernel(const KernelParams params)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t pixelCount = params.frame.width * params.frame.height;
    if (pixel >= pixelCount || params.psfGatherBuckets == nullptr || params.psfBinCount == 0)
        return;

    const Sensor& sensor = params.scene.camera->Dispatch([](const auto* camera) -> const Sensor& {
        return camera->getSensor();
    });
    const GatherPsfSensor* gatherSensor = sensor.CastOrNullptr<GatherPsfSensor>();
    if (gatherSensor == nullptr)
        return;

    const int destX = static_cast<int>(pixel % params.frame.width);
    const int destY = static_cast<int>(pixel / params.frame.width);
    const glm::vec4 out = gatherSensor->resolvePixel(
        pixel, params.frame.width, params.frame.height,
        params.psfBinCount, params.psfGatherBuckets);
    params.accumulation[pixel] = out;
    surf2Dwrite(make_float4(out.x, out.y, out.z, out.w),
        params.output.color, destX * sizeof(float4), destY);
}
