#include <cuda_fp16.h>

#include "Raytracing/KernelHelpers.h"
#include "Samplers/RandomSampler.h"

// Deterministic, single-ray-per-pixel primary ray for the AOV pass: pixel-center
// (no AA jitter), no depth-of-field (centered lens/pupil sample). Densely indexed by
// pixel — no compaction, since every pixel always produces exactly one ray.
NR_GPU_KERNEL void generateAovKernel(const KernelParams params)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t total = params.frame.width * params.frame.height;
    if (pixel >= total)
        return;

    const uint32_t x = pixel % params.frame.width;
    const uint32_t y = pixel / params.frame.width;
    const float nx = (static_cast<float>(x) + 0.5f) / static_cast<float>(params.frame.width) * 2.0f - 1.0f;
    const float ny = 1.0f - (static_cast<float>(y) + 0.5f) / static_cast<float>(params.frame.height) * 2.0f;

    glm::vec3 origin{};
    glm::vec3 direction{};
    float cameraWeight = 1.0f;
    RandomState rng = seedRandom(static_cast<uint64_t>(pixel));
    const bool active = params.scene.camera->Dispatch([&](const auto* camera) {
        return camera->generateRay(origin, direction, cameraWeight, nx, ny, rng,
            pixel, 550.0f, true);
    });

    PathRayWorkItem ray{};
    ray.origin = origin;
    ray.direction = active ? direction : glm::vec3(0.0f, 0.0f, -1.0f);
    ray.sampleIndex = pixel;
    params.queues.aovRayQueue[pixel] = ray;
}
