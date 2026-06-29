#include <cuda_fp16.h>

#include "Raytracing/KernelHelpers.h"
#include "Raytracing/Queues.h"
#include "Samplers/RandomSampler.h"

NR_GPU_KERNEL void generateKernel(const KernelParams params)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t total = params.frame.width * params.frame.height;
    const bool inRange = pixel < total;
    bool active = inRange && !adaptiveSkip(params, pixel);
    uint32_t rng = pcgHash(pixel ^ (params.frame.totalAccumulated * 1664525u + 1013904223u));
    glm::vec3 origin{};
    glm::vec3 direction{};
    float cameraWeight = 1.0f;
    if (active)
    {
        const uint32_t x = pixel % params.frame.width;
        const uint32_t y = pixel / params.frame.width;
        const glm::vec2 jitter = params.frame.sampleIndex == 0 && params.frame.frame == 0
            ? glm::vec2(0.5f)
            : ActiveSampler::sample(params.frame.totalAccumulated);
        const float nx = (static_cast<float>(x) + jitter.x) / static_cast<float>(params.frame.width) * 2.0f - 1.0f;
        const float ny = 1.0f - (static_cast<float>(y) + jitter.y) / static_cast<float>(params.frame.height) * 2.0f;
        active = params.scene.camera->Dispatch([&](const auto* camera) {
            return camera->generateRay(origin, direction, cameraWeight, nx, ny, rng,
                pixel);
        });
    }

    PathRayWorkItem ray{};
    if (active)
    {
        PathState state{};
        state.throughput = glm::vec3(cameraWeight);
        state.rngState = rng == 0 ? 1 : rng;
        params.queues.pathStates[pixel] = state;
        PrimaryState primary{};
        primary.primaryObjectIndex = InvalidIndex;
        params.queues.primaryStates[pixel] = primary;
        ray.origin = origin;
        ray.direction = direction;
        ray.sampleIndex = pixel;
    }
    appendRayWarp(params.queues, 0, active, ray);
}
