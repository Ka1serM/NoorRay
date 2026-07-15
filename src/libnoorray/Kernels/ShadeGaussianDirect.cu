#include "Raytracing/GaussianShading.h"
#include "Raytracing/RgbToSpectrum.h"
#include "Raytracing/SceneData.h"

// Specialized terminal shader for Gaussian-only direct-color rendering. It
// deliberately excludes mesh materials, GI, shadow work, and training so the
// common viewport path does not inherit their register and stack footprint.
NR_GPU_KERNEL void shadeGaussianDirectKernel(const KernelParams params)
{
    const uint32_t index = NR_GPU_LAUNCH_IDX;
    const uint32_t activeCount = params.queues.rayCounts[params.depth];
    if (index >= activeCount)
        return;

    const HitWorkItem hit = params.queues.hitQueue[index];
    PathState state = params.queues.pathStates[hit.sampleIndex];
    if (hit.instanceIndex != InvalidIndex)
    {
        const uint32_t gaussianId = hit.instanceIndex;
        const glm::vec3* shCoeffs = params.scene.gaussianShCoeffs + gaussianId * 16;
        glm::vec3 rgb = glm::vec3(0.5f) + evaluateGaussianSphericalHarmonics(
            shCoeffs,
            params.scene.renderSettings.gaussianRenderSphericalHarmonics,
            -hit.direction);
        rgb.x = fminf(fmaxf(rgb.x, 0.0f), 1.0f);
        rgb.y = fminf(fmaxf(rgb.y, 0.0f), 1.0f);
        rgb.z = fminf(fmaxf(rgb.z, 0.0f), 1.0f);
        const SampledSpectrum albedo = rgbAlbedoToSpectrumTexture(
            rgb, state.wl, params.scene.spectrumTableScale,
            params.scene.spectrumTableTexture);
        state.radiance += state.throughput * albedo;
        state.packedCounters |= 1u << CounterHitShift;
        state.packedCounters += 1u << CounterDiffuseShift;
        params.queues.pathStates[hit.sampleIndex] = state;
        return;
    }

    const bool backgroundVisible = params.scene.environment->visible != 0
        && !params.scene.renderSettings.transparentBackground;
    if (!backgroundVisible)
        return;

    const glm::vec3 rgb = params.scene.environment->rgbRadiance(
        params.scene.textures, params.scene.textureCount, hit.direction, true);
    state.radiance += state.throughput * rgbIlluminantToSpectrumTexture(
        rgb, state.wl, params.scene.spectrumTableScale,
        params.scene.spectrumTableTexture, params.scene.d65);
    state.packedCounters |= 1u << CounterHitShift;
    params.queues.pathStates[hit.sampleIndex] = state;
}
