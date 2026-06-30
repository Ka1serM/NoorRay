#pragma once

#include <cuda_fp16.h>

#include "Raytracing/RgbToSpectrum.h"
#include "Raytracing/SceneData.h"
#include "Samplers/HemisphereSampler.h"
#include "Samplers/RandomSampler.h"

static constexpr float Pi = 3.14159265358979323846f;
static constexpr uint32_t CounterDiffuseShift = 0;
static constexpr uint32_t CounterSpecularShift = 8;
static constexpr uint32_t CounterTransmissionShift = 16;
static constexpr uint32_t CounterHitShift = 24;

struct EnvironmentDirectionSample
{
    glm::vec3 direction{};
    float pdf{};
};

NR_GPU inline float powerHeuristic(const float a, const float b)
{
    const float a2 = a * a;
    const float b2 = b * b;
    return a2 / fmaxf(a2 + b2, 1e-20f);
}

NR_GPU inline glm::vec2 environmentUv(const Environment& environment, const glm::vec3 direction)
{
    const float rotatedX = environment.rotationCos * direction.x
        - environment.rotationSin * direction.z;
    const float rotatedZ = environment.rotationSin * direction.x
        + environment.rotationCos * direction.z;
    return {
        0.5f + atan2f(rotatedZ, rotatedX) / (2.0f * Pi),
        acosf(fminf(fmaxf(direction.y, -1.0f), 1.0f)) / Pi};
}

NR_GPU inline float4 environmentCdfTexel(const Environment& environment, int x, int y)
{
    x = max(0, min(x, environment.cdfWidth - 1));
    y = max(0, min(y, environment.cdfHeight - 1));
    return tex2D<float4>(environment.cdfTexture,
        (static_cast<float>(x) + 0.5f) / static_cast<float>(environment.cdfWidth),
        (static_cast<float>(y) + 0.5f) / static_cast<float>(environment.cdfHeight));
}

NR_GPU inline float environmentPdf(const Environment& environment, const glm::vec3 direction)
{
    if (environment.cdfTexture == 0 || environment.cdfWidth <= 0 || environment.cdfHeight <= 0)
        return 1.0f / (4.0f * Pi);
    const glm::vec2 uv = environmentUv(environment, direction);
    const int x = min(static_cast<int>(uv.x * environment.cdfWidth), environment.cdfWidth - 1);
    const int y = min(static_cast<int>(uv.y * environment.cdfHeight), environment.cdfHeight - 1);
    const float rowCdf = environmentCdfTexel(environment, 0, y).y;
    const float previousRowCdf = y > 0 ? environmentCdfTexel(environment, 0, y - 1).y : 0.0f;
    const float columnCdf = environmentCdfTexel(environment, x, y).x;
    const float previousColumnCdf = x > 0 ? environmentCdfTexel(environment, x - 1, y).x : 0.0f;
    const float probability = fmaxf(rowCdf - previousRowCdf, 0.0f)
        * fmaxf(columnCdf - previousColumnCdf, 0.0f);
    const float theta0 = Pi * static_cast<float>(y) / static_cast<float>(environment.cdfHeight);
    const float theta1 = Pi * static_cast<float>(y + 1) / static_cast<float>(environment.cdfHeight);
    const float solidAngle = (2.0f * Pi / static_cast<float>(environment.cdfWidth))
        * fmaxf(cosf(theta0) - cosf(theta1), 1e-12f);
    return probability / solidAngle;
}

NR_GPU inline EnvironmentDirectionSample sampleEnvironmentDirection(
    const Environment& environment, RandomState& rng)
{
    EnvironmentDirectionSample sample{};
    if (environment.cdfTexture == 0 || environment.cdfWidth <= 0 || environment.cdfHeight <= 0) {
        const float z = 1.0f - 2.0f * randomFloat(rng);
        const float phi = 2.0f * Pi * randomFloat(rng);
        const float radius = sqrtf(fmaxf(1.0f - z * z, 0.0f));
        sample.direction = glm::vec3(radius * cosf(phi), z, radius * sinf(phi));
        sample.pdf = 1.0f / (4.0f * Pi);
        return sample;
    }

    const float marginalSample = randomFloat(rng);
    int low = 0, high = environment.cdfHeight - 1;
    while (low < high) {
        const int mid = (low + high) / 2;
        if (environmentCdfTexel(environment, 0, mid).y < marginalSample)
            low = mid + 1;
        else
            high = mid;
    }
    const int y = low;
    const float conditionalSample = randomFloat(rng);
    low = 0; high = environment.cdfWidth - 1;
    while (low < high) {
        const int mid = (low + high) / 2;
        if (environmentCdfTexel(environment, mid, y).x < conditionalSample)
            low = mid + 1;
        else
            high = mid;
    }
    const int x = low;

    const float phi = 2.0f * Pi * ((static_cast<float>(x) + randomFloat(rng))
        / static_cast<float>(environment.cdfWidth) - 0.5f);
    const float theta0 = Pi * static_cast<float>(y) / static_cast<float>(environment.cdfHeight);
    const float theta1 = Pi * static_cast<float>(y + 1) / static_cast<float>(environment.cdfHeight);
    const float cosTheta = cosf(theta0)
        + randomFloat(rng) * (cosf(theta1) - cosf(theta0));
    const float sinTheta = sqrtf(fmaxf(1.0f - cosTheta * cosTheta, 0.0f));
    const glm::vec3 rotated(sinTheta * cosf(phi), cosTheta, sinTheta * sinf(phi));
    sample.direction = glm::normalize(glm::vec3(
        environment.rotationCos * rotated.x + environment.rotationSin * rotated.z,
        rotated.y,
        -environment.rotationSin * rotated.x + environment.rotationCos * rotated.z));
    sample.pdf = environmentPdf(environment, sample.direction);
    return sample;
}

NR_GPU inline SampledSpectrum environmentRadiance(
    const GpuSceneData scene,
    const glm::vec3 direction,
    const bool cameraRay,
    const SampledWavelengths& wl)
{
    const Environment& environment = *scene.environment;
    glm::vec3 rgb = environment.color;
    if (environment.textureIndex >= 0 &&
        static_cast<uint32_t>(environment.textureIndex) < scene.textureCount)
    {
        const glm::vec2 uv = environmentUv(environment, direction);
        const glm::vec4 textureColor = scene.textures[environment.textureIndex].sample(uv);
        rgb *= glm::vec3(textureColor);
    }
    rgb *= cameraRay ? environment.visibleExposureScale : environment.lightingExposureScale;
    return rgbIlluminantToSpectrum(
        rgb, wl, scene.spectrumTableScale, scene.spectrumTableCoeffs, scene.d65);
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
