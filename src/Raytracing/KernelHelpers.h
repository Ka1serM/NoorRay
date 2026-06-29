#pragma once

#include <cuda_fp16.h>

#include "Raytracing/SceneData.h"
#include "Samplers/RandomSampler.h"

static constexpr float Pi = 3.14159265358979323846f;
static constexpr uint32_t CounterDiffuseShift = 0;
static constexpr uint32_t CounterSpecularShift = 8;
static constexpr uint32_t CounterTransmissionShift = 16;
static constexpr uint32_t CounterHitShift = 24;

NR_GPU inline glm::vec3 environmentRadiance(
    const GpuSceneData scene,
    const glm::vec3 direction,
    const bool cameraRay)
{
    const Environment& environment = *scene.environment;
    glm::vec3 radiance = environment.color;
    if (environment.textureIndex >= 0 &&
        static_cast<uint32_t>(environment.textureIndex) < scene.textureCount)
    {
        const float rotatedX = environment.rotationCos * direction.x - environment.rotationSin * direction.z;
        const float rotatedZ = environment.rotationSin * direction.x + environment.rotationCos * direction.z;
        const float u = 0.5f + atan2f(rotatedZ, rotatedX) / (2.0f * Pi);
        const float v = acosf(fminf(fmaxf(direction.y, -1.0f), 1.0f)) / Pi;
        const glm::vec4 textureColor = scene.textures[environment.textureIndex].sample({u, v});
        radiance *= glm::vec3(textureColor);
    }
    return radiance * (cameraRay
        ? environment.visibleExposureScale
        : environment.lightingExposureScale);
}

NR_GPU inline glm::vec3 cosineHemisphere(const glm::vec3 normal, uint32_t& rng)
{
    const float r1 = randomFloat(rng);
    const float r2 = randomFloat(rng);
    const float radius = sqrtf(r1);
    const float angle = 2.0f * Pi * r2;
    const glm::vec3 helper = fabsf(normal.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    const glm::vec3 tangent = glm::normalize(glm::cross(helper, normal));
    const glm::vec3 bitangent = glm::cross(normal, tangent);
    return glm::normalize(tangent * (radius * cosf(angle)) +
                           bitangent * (radius * sinf(angle)) +
                           normal * sqrtf(fmaxf(0.0f, 1.0f - r1)));
}

NR_GPU inline ushort4 packHalf4(const glm::vec3 value, const float w)
{
    return make_ushort4(
        __half_as_ushort(__float2half(value.x)),
        __half_as_ushort(__float2half(value.y)),
        __half_as_ushort(__float2half(value.z)),
        __half_as_ushort(__float2half(w)));
}

NR_GPU inline bool adaptiveSkip(const KernelParams params, const uint32_t pixel)
{
    const RenderSettings settings = params.scene.renderSettings;
    if (settings.adaptiveSamplingEnabled == 0 || params.frame.totalAccumulated == 0)
        return false;
    const glm::vec4 state = params.adaptiveState[pixel];
    if (state.z < static_cast<float>(settings.adaptiveMinSamples) || state.z < 2.0f)
        return false;
    const float variance = state.y / fmaxf(state.z - 1.0f, 1.0f);
    const float error = sqrtf(fmaxf(variance, 0.0f) / state.z) / fmaxf(fabsf(state.x), 1e-4f);
    return error <= settings.adaptiveTargetError;
}
