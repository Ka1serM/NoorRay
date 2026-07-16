#include "Raytracing/Queues.h"
#include "Samplers/OwenSobolSampler.h"
#include "Samplers/RandomSampler.h"
#include "Training/GaussianTrainData.h"

NR_GPU_KERNEL void generateGaussianTrainKernel(const GaussianTrainingKernelParams params)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t total = params.frame.width * params.frame.height;
    const bool active = pixel < total;
    PathRayWorkItem ray{};
    if (active)
    {
        const uint32_t x = pixel % params.frame.width;
        const uint32_t y = pixel / params.frame.width;
        const OwenSobolSampler sampler({
            params.frame.totalAccumulated, hashCombine32(x, y)});
        const SampledWavelengths wl = SampledWavelengths::sampleVisible(
            sampler.sample1D(SampleDimension::Wavelength));
        glm::vec2 jitter(0.5f, 0.5f);
        if (params.frame.frameIndex != 0)
            jitter = sampler.sample2D(PixelSampleDimensions);

        ray.origin = glm::vec3(params.train.cameraToWorld[3]);
        const glm::vec3 cameraDirection(
            (static_cast<float>(x) + jitter.x - params.train.cx) / params.train.fx,
            -(static_cast<float>(y) + jitter.y - params.train.cy) / params.train.fy,
            -1.0f);
        ray.direction = glm::normalize(glm::vec3(
            params.train.cameraToWorld * glm::vec4(cameraDirection, 0.0f)));
        ray.sampleIndex = pixel;

        const uint64_t sequence =
            (static_cast<uint64_t>(params.frame.totalAccumulated) << 32u) | pixel;
        PathState state{};
        state.wl = wl;
        state.throughput = SampledSpectrum(1.0f);
        state.rngState = seedRandom(sequence);
        state.etaScale = 1.0f;
        state.cameraWeight = 1.0f;
        params.queues.pathStates[pixel] = state;
    }
    appendRayWarp(params.queues, 0, active, ray);
}

NR_GPU_KERNEL void shadeGaussianTrainForwardKernel(const GaussianTrainingKernelParams params)
{
    const uint32_t index = NR_GPU_LAUNCH_IDX;
    const uint32_t activeCount = params.queues.rayCounts[params.depth];
    if (index >= activeCount)
        return;

    const HitWorkItem hit = params.queues.hitQueue[index];
    const bool isGaussian = hit.instanceIndex != InvalidIndex
        && hit.instanceIndex >= params.scene.meshInstanceCount;
    if (!isGaussian)
        return;

    const uint32_t gaussianId = hit.instanceIndex - params.scene.meshInstanceCount;
    PathState state = params.queues.pathStates[hit.sampleIndex];
    state.trainColor += state.cameraWeight * params.train.colorRgb[gaussianId];
    params.queues.pathStates[hit.sampleIndex] = state;
}

NR_GPU_KERNEL void finalizeGaussianTrainForwardKernel(const GaussianTrainingKernelParams params)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t pixelCount = params.frame.width * params.frame.height;
    if (pixel >= pixelCount)
        return;

    const glm::vec3 color = params.queues.pathStates[pixel].trainColor
        / static_cast<float>(params.train.samplesPerPixel);
    atomicAdd(&params.train.outputColor[pixel].x, color.x);
    atomicAdd(&params.train.outputColor[pixel].y, color.y);
    atomicAdd(&params.train.outputColor[pixel].z, color.z);
}
