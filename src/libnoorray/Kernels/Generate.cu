#include <cuda_fp16.h>

#include "Raytracing/Queues.h"
#include "Raytracing/SceneData.h"
#include "Samplers/OwenSobolSampler.h"
#include "Samplers/RandomSampler.h"

NR_GPU_KERNEL void generateKernel(const KernelParams params)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t total = params.frame.width * params.frame.height;
    const bool inRange = pixel < total;
    bool active = inRange;
    const uint64_t sequence = (static_cast<uint64_t>(params.frame.totalAccumulated) << 32u)
        | static_cast<uint64_t>(pixel);
    RandomState rng = seedRandom(sequence);
    glm::vec3 origin{};
    glm::vec3 direction{};
    float cameraWeight = 1.0f;
    SampledWavelengths wl{};
    if (active)
    {
        const OwenSobolSampler sampler({params.frame.totalAccumulated, pixel});
        const float wavelengthSample = sampler.sample1D(SampleDimension::Wavelength);
        wl = SampledWavelengths::sampleVisible(wavelengthSample);
        const uint32_t x = pixel % params.frame.width;
        const uint32_t y = pixel / params.frame.width;
        // Per-pixel Owen scrambling decorrelates neighboring pixels while each
        // pixel retains a low-discrepancy sequence over accumulated samples.
        // While the camera is being moved, freeze the sample at the pixel
        // center instead: a jittered live-preview reads as shimmer/noise
        // during motion, and there's no accumulation to converge it anyway.
        // frame 0 covers both the very first frame and every frame while
        // resetAccumulation keeps re-triggering (continuous camera drag/orbit/fly).
        const bool cameraMoving = params.frame.frameIndex == 0;
        glm::vec2 jitter(0.5f, 0.5f);
        if (!cameraMoving)
        {
            jitter = sampler.sample2D(PixelSampleDimensions);
        }
        const glm::vec2 lensSample = sampler.sample2D(LensSampleDimensions);
        const float nx = (static_cast<float>(x) + jitter.x) / static_cast<float>(params.frame.width) * 2.0f - 1.0f;
        const float ny = 1.0f - (static_cast<float>(y) + jitter.y) / static_cast<float>(params.frame.height) * 2.0f;
        active = params.scene.camera->Dispatch([&](const auto* camera) {
            return camera->generateRay(origin, direction, cameraWeight, nx, ny, lensSample,
                pixel, wl);
        });

        // Camera samples rejected by the lens still represent a black sample.
        // Initialize their state so finalization does not reuse the previous path.
        if (!active)
        {
            PathState state{};
            state.wl = wl;
            if (params.scene.camera->Is<RealisticCamera>() || params.scene.camera->Is<HybridPsfCamera>())
                state.packedCounters |= 1u << CounterHitShift;
            state.rngState = rng;
            params.queues.pathStates[pixel] = state;
        }
    }

    PathRayWorkItem ray{};
    if (active)
    {
        PathState state{};
        state.wl = wl;

        state.throughput = SampledSpectrum(cameraWeight);
        state.radiance   = SampledSpectrum(0.f);
        state.rngState   = rng;
        state.etaScale   = 1.0f;
        state.cameraWeight = cameraWeight;

        params.queues.pathStates[pixel] = state;
        ray.origin      = origin;
        ray.direction   = direction;
        ray.sampleIndex = pixel;
    }
    appendRayWarp(params.queues, 0, active, ray);
}

NR_GPU_KERNEL void generateGaussianDirectKernel(const KernelParams params)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t total = params.frame.width * params.frame.height;
    if (pixel >= total)
        return;

    const OwenSobolSampler sampler({params.frame.totalAccumulated, pixel});
    const float wavelengthSample = sampler.sample1D(SampleDimension::Wavelength);
    SampledWavelengths wl = params.scene.camera->Is<RealisticCamera>()
        ? SampledWavelengths::sampleVisibleSingle(wavelengthSample)
        : SampledWavelengths::sampleVisible(wavelengthSample);
    const uint32_t x = pixel % params.frame.width;
    const uint32_t y = pixel / params.frame.width;
    glm::vec2 jitter(0.5f);
    if (params.frame.frameIndex != 0)
        jitter = sampler.sample2D(PixelSampleDimensions);

    glm::vec3 origin{};
    glm::vec3 direction{};
    float cameraWeight = 1.0f;
    const float nx = (static_cast<float>(x) + jitter.x)
        / static_cast<float>(params.frame.width) * 2.0f - 1.0f;
    const float ny = 1.0f - (static_cast<float>(y) + jitter.y)
        / static_cast<float>(params.frame.height) * 2.0f;
    const bool active = params.scene.camera->Dispatch([&](const auto* camera) {
        return camera->generateRay(origin, direction, cameraWeight, nx, ny,
            sampler.sample2D(LensSampleDimensions), pixel, wl);
    });

    PathState state{};
    state.wl = wl;
    state.throughput = SampledSpectrum(cameraWeight);
    state.cameraWeight = cameraWeight;
    if (!active)
        state.packedCounters |= 1u << CounterHitShift;
    params.queues.pathStates[pixel] = state;

    PathRayWorkItem ray{};
    ray.origin = origin;
    ray.direction = direction;
    ray.sampleIndex = pixel;
    appendRayWarp(params.queues, 0, active, ray);
}
