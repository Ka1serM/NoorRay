#pragma once

#include <cstdint>

#include "Raytracing/SceneData.h"
#include "Samplers/RandomSampler.h"

struct AnalyticLightAliasEntry
{
    float threshold{};
    uint32_t alias{};
    float selectionPdf{};
};

struct AnalyticLightSample
{
    LightSample light{};
    float selectionPdf{};
};

#if defined(NR_GPU_CODE)
NR_GPU inline AnalyticLightSample sampleAnalyticLight(
    const GpuSceneData& scene,
    const glm::vec3 position,
    RandomState& rng,
    const SampledWavelengths& wl)
{
    AnalyticLightSample result{};
    if (scene.analyticLightAliasCount == 0 || scene.analyticLightAliases == nullptr)
        return result;

    const float tableSample = randomFloat(rng) * scene.analyticLightAliasCount;
    const uint32_t column = min(
        static_cast<uint32_t>(tableSample), scene.analyticLightAliasCount - 1);
    const AnalyticLightAliasEntry columnEntry = scene.analyticLightAliases[column];
    const uint32_t selected = tableSample - static_cast<float>(column)
            < columnEntry.threshold
        ? column
        : columnEntry.alias;
    const AnalyticLightAliasEntry selectedEntry = scene.analyticLightAliases[selected];
    result.selectionPdf = selectedEntry.selectionPdf;

    uint32_t localIndex = selected;
    if (localIndex < scene.pointLightCount)
    {
        result.light = scene.pointLights[localIndex].sampleLi(
            position, rng, wl, scene.spectrumTableScale,
            scene.spectrumTableCoeffs, scene.d65);
        return result;
    }
    localIndex -= scene.pointLightCount;
    if (localIndex < scene.spotLightCount)
    {
        result.light = scene.spotLights[localIndex].sampleLi(
            position, rng, wl, scene.spectrumTableScale,
            scene.spectrumTableCoeffs, scene.d65);
        return result;
    }
    localIndex -= scene.spotLightCount;
    if (localIndex < scene.rectLightCount)
    {
        result.light = scene.rectLights[localIndex].sampleLi(
            position, rng, wl, scene.spectrumTableScale,
            scene.spectrumTableCoeffs, scene.d65);
        return result;
    }
    localIndex -= scene.rectLightCount;
    if (localIndex < scene.directionalLightCount)
    {
        result.light = scene.directionalLights[localIndex].sampleLi(
            position, rng, wl, scene.spectrumTableScale,
            scene.spectrumTableCoeffs, scene.d65);
    }
    return result;
}
#endif
