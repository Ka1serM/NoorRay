#pragma once

#include <optix_device.h>

#include "Backend/OptiX/ABI/Geometry.h"
#include "Rendering/MisHeuristic.h"

extern "C"
{
extern __constant__ KernelParams params;
}

#include "Rendering/Lighting/DirectLightSampling.h"

namespace nr::path_miss
{

NR_GPU inline SampledSpectrum environmentRadiance(
    const Ray& ray, const PathState& state)
{
    const bool cameraRay = state.depth == 0;
    float misWeight = 1.0f;
    if (!cameraRay)
    {
        const float environmentWeight = fmaxf(
            params.scene.environment->importanceWeight, 0.0f);
        if (state.lastBsdfPdf > 0.0f && environmentWeight > 0.0f
            && nr::direct_light::environmentLightMixtureProbability() > 0.0f)
        {
            misWeight = powerHeuristic(state.lastBsdfPdf,
                nr::direct_light::lightPdf(
                    params.scene.environment->pdf(ray.direction())));
        }
    }
    return params.scene.environment->radiance(
        params.scene.textures, params.scene.textureCount, ray.direction(), cameraRay,
        state.wl, params.scene.spectrumTableScale, params.scene.spectrumTableCoeffs,
        params.scene.d65) * misWeight;
}

}

extern "C" __global__ void __miss__pathTrace()
{
    PathTracePayload* const payload = getPathTracePayload<>();
    PathState& state = *payload->state;
    const bool cameraPath = state.depth == 0;
    const bool backgroundVisible =
        !params.scene.renderSettings.transparentBackground;
    if (!cameraPath || backgroundVisible)
    {
        const SampledSpectrum contribution = state.throughput
            * nr::path_miss::environmentRadiance(payload->ray, state);
        state.radiance += nr::lighting::clampIndirectLightContribution(
            contribution, params.scene.renderSettings.indirectLightClamp,
            state.depth);
    }
    if (cameraPath && backgroundVisible)
        state.alpha = 1.0f;
    payload->status = PathTraceStatus::Terminate;
}
