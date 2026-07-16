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
    // Gaussians occlude shadow rays the same way they occlude camera/bounce
    // rays: the any-hit in GaussianHit.cu already runs the unbiased
    // accept-with-probability-alpha test (Sun et al., "Stochastic Ray Tracing
    // of Transparent 3D Gaussians"), so a returned Gaussian hit here just
    // means "already confirmed opaque enough" — no separate opacity test
    // needed, unlike mesh materials below.
    const bool gaussianEnabled = params.scene.gaussianCount > 0;
    const bool meshVisibilityBoundEnabled = gaussianEnabled && params.scene.meshInstanceCount > 0;
    float rayMin = shadow.tMin;
    while (rayMin < shadow.tMax)
    {
        // Fresh sample index per gaussian any-hit RR draw so shadow rays
        // don't reuse (and correlate with) the same draws as the primary/
        // bounce ray that reached this shading point.
        const uint32_t gaussianSampleIndex = gaussianEnabled ? randomUint(rng) : 0;
        const RayHit hit = intersectRay(params.scene.tlasHandle, shadow.origin,
            shadow.direction, rayMin, shadow.tMax, gaussianSampleIndex,
            gaussianEnabled, meshVisibilityBoundEnabled,
            shadow.excludedGaussianId, true);
        if (hit.instanceIndex == InvalidIndex)
            return false;

        if (hit.primitiveIndex == InvalidIndex)
            return true;

        const SurfaceMaterialData surface = loadSurfaceMaterial(params.scene, hit.instanceIndex,
            hit.primitiveIndex, hit.u, hit.v);

        float opacity = surface.material->opacity;
        if (surface.material->opacityIndex >= 0)
            opacity *= params.scene.textures[surface.material->opacityIndex]
                .sample(surface.uv).w;
        opacity = fminf(fmaxf(opacity, 0.0f), 1.0f);

        // Glass (transmission > 0) should not cast the same hard shadow as an
        // opaque surface: only the non-transmissive fraction of the (present,
        // per `opacity`) surface actually blocks the ray. This ignores the
        // refraction bending a real shadow ray would undergo — the same
        // straight-through approximation already used for NEE generally —
        // but at least stops fully transmissive glass from reading as opaque.
        float transmission = surface.material->transmission;
        if (surface.material->transmissionIndex >= 0)
            transmission *= params.scene.textures[surface.material->transmissionIndex]
                .sample(surface.uv).x;
        transmission = fminf(fmaxf(transmission, 0.0f), 1.0f);

        const float blockProbability = opacity * (1.0f - transmission);
        if (blockProbability >= 1.0f || randomFloat(rng) < blockProbability)
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

    if (params.queues.shadowQueue[index].tMax <= 0.0f)
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
