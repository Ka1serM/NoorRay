#include <cuda_fp16.h>

#include "Raytracing/Queues.h"
#include "Raytracing/SceneData.h"
#include "Samplers/RandomSampler.h"
#include "Samplers/R2Sampler.h"

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
        uint32_t wavelengthRng = pcgHash(pixel ^ 0x68bc21ebu);
        float wavelengthSample = randomFloat(wavelengthRng)
            + 0.61803398875f * static_cast<float>(params.frame.totalAccumulated);
        wavelengthSample -= floorf(wavelengthSample);
        wl = SampledWavelengths::sampleVisible(wavelengthSample);
        const uint32_t x = pixel % params.frame.width;
        const uint32_t y = pixel / params.frame.width;
        // A fixed random rotation decorrelates neighboring pixels while each
        // pixel retains the low-discrepancy R2 sequence over accumulated samples.
        uint32_t jitterRng = pcgHash(pixel ^ 0x9e3779b9u);
        const glm::vec2 rotation(randomFloat(jitterRng), randomFloat(jitterRng));
        glm::vec2 jitter = R2Sampler::sample(params.frame.totalAccumulated) + rotation;
        jitter.x -= floorf(jitter.x);
        jitter.y -= floorf(jitter.y);
        const float nx = (static_cast<float>(x) + jitter.x) / static_cast<float>(params.frame.width) * 2.0f - 1.0f;
        const float ny = 1.0f - (static_cast<float>(y) + jitter.y) / static_cast<float>(params.frame.height) * 2.0f;
        active = params.scene.camera->Dispatch([&](const auto* camera) {
            return camera->generateRay(origin, direction, cameraWeight, nx, ny, rng,
                pixel, wl);
        });

        // Camera samples rejected by the lens still represent a black sample.
        // Initialize their state so finalization does not reuse the previous path.
        if (!active)
        {
            PathState state{};
            state.wl = wl;
            if (params.scene.camera->Is<RealisticCamera>() || params.scene.camera->Is<RossPsfCamera>())
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

        params.queues.pathStates[pixel] = state;
        ray.origin      = origin;
        ray.direction   = direction;
        ray.sampleIndex = pixel;
    }
    appendRayWarp(params.queues, 0, active, ray);
}
