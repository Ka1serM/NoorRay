#ifdef NR_OPTIX_PTX_BUILD

#include <optix_device.h>

#include "Raytracing/Geometry.h"
#include "Raytracing/RayTraversal.h"
#include "Samplers/RandomSampler.h"

extern "C"
{
extern __constant__ KernelParams params;
}

NR_GPU inline bool shadowOccluded(
    const ShadowWorkItem& shadow, RandomState& rng)
{
    float rayMin = shadow.tMin;
    while (rayMin < shadow.tMax)
    {
        const RayHit hit = intersectRay(params.scene.tlasHandle, shadow.origin,
            shadow.direction, rayMin, shadow.tMax, 0, false);
        if (!hit.hit)
            return false;

        const SurfaceData surface = loadSurface(params.scene, hit.instanceIndex,
            hit.primitiveIndex, hit.u, hit.v);
        if (surface.material == nullptr)
            return true;

        float opacity = surface.material->opacity;
        if (surface.material->opacityIndex >= 0)
            opacity *= params.scene.textures[surface.material->opacityIndex]
                .sample(surface.uv).w;
        opacity = fminf(fmaxf(opacity, 0.0f), 1.0f);
        if (opacity >= 1.0f || (opacity > 0.0f && randomFloat(rng) < opacity))
            return true;

        // OptiX reports t in the original ray parameterization. Advance past
        // this primitive while retaining the original finite-light endpoint.
        rayMin = hit.t + fmaxf(1e-4f, fabsf(hit.t) * 1e-6f);
    }
    return false;
}

extern "C" __global__ void __raygen__connect()
{
    const uint32_t index = NR_GPU_OPTIX_LAUNCH_ID;
    const uint32_t activeCount = params.queues.rayCounts[params.depth];
    if (index >= activeCount)
        return;

    const ShadowWorkItem shadow = params.queues.shadowQueue[index];
    if (shadow.tMax <= shadow.tMin)
        return;
    PathState& state = params.queues.pathStates[shadow.sampleIndex];
    RandomState rng = shadow.rngState;
    if (!shadowOccluded(shadow, rng))
        state.radiance += shadow.contribution;
}

#endif
