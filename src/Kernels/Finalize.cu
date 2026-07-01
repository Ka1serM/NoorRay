#include <cuda_fp16.h>

#include "Raytracing/KernelHelpers.h"

NR_GPU_KERNEL void finalizeKernel(const KernelParams params)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t pixelCount = params.frame.width * params.frame.height;
    if (pixel >= pixelCount || adaptiveSkip(params, pixel))
        return;

    const PathState state = params.queues.pathStates[pixel];
    const float alpha = static_cast<float>((state.packedCounters >> CounterHitShift) & 1u);

    // Reconstruct wavelengths for spectral→XYZ conversion.
    SampledWavelengths wl;
    for (int i = 0; i < NrSpectrumSamples; ++i)
    {
        wl.lambda[i] = state.lambda[i];
        wl.pdf[i]    = state.lambdaPdf[i];
    }

    // Spectral radiance → linear sRGB via CIE XYZ.
    const glm::vec3 xyz = spectrumToXYZ(
        state.radiance, wl, params.scene.cieX, params.scene.cieY, params.scene.cieZ);
    glm::vec3 radiance = xyzToLinearSRGB(xyz);
    const glm::vec3 sampleRadiance = radiance;

    const uint32_t x = pixel % params.frame.width;
    const uint32_t y = pixel / params.frame.width;
    const float weight = params.frame.totalAccumulated == 0
        ? 1.0f : 1.0f / static_cast<float>(params.frame.totalAccumulated + 1);
    const glm::vec4 previous = params.accumulation[pixel];
    radiance = glm::mix(glm::vec3(previous.x, previous.y, previous.z), radiance, weight);
    const float accAlpha = previous.w * (1.0f - weight) + alpha * weight;
    params.accumulation[pixel] = glm::vec4(radiance, accAlpha);
    surf2Dwrite(make_float4(radiance.x, radiance.y, radiance.z, accAlpha),
        params.output.color, x * sizeof(float4), y);

    // Luminance from linear sRGB for adaptive sampling statistics.
    const float luminance = sampleRadiance.x * 0.2126f
        + sampleRadiance.y * 0.7152f + sampleRadiance.z * 0.0722f;
    float mean = luminance;
    float m2   = 0.0f;
    float count = 1.0f;
    if (params.frame.totalAccumulated > 0)
    {
        const glm::vec4 old = params.adaptiveState[pixel];
        count = old.z + 1.0f;
        const float delta = luminance - old.x;
        mean = old.x + delta / count;
        m2   = old.y + delta * (luminance - mean);
    }
    params.adaptiveState[pixel] = glm::vec4(mean, m2, count, 0.0f);
}
