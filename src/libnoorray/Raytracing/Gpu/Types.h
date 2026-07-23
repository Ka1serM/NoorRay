#pragma once

#include <cstdint>

#include "CUDA/Annotations.h"
#include "Raytracing/Ray.h"
#include "Shading/Spectrum.h"
#include "Samplers/RandomSampler.h"

static constexpr uint32_t InvalidIndex = ~0u;
// AOV Cryptomatte queries reserve this sample value to request analytic,
// full-opacity Gaussian intersections. It is only interpreted while an AOV
// launch is active, so regular path samples remain unaffected.
static constexpr uint32_t OpaqueAovGaussianSample = InvalidIndex - 1u;
static constexpr uint8_t MeshVisibility = uint8_t{1} << 0u;
static constexpr uint8_t GaussianVisibility = uint8_t{1} << 1u;
static constexpr uint8_t SceneVisibility = MeshVisibility | GaussianVisibility;
static constexpr uint8_t FullVisibility = ~uint8_t{0};

using TlasHandle = uint64_t;

// POD result of an OptiX scene query. Mesh hits carry barycentrics; Gaussian
// hits leave primitiveIndex invalid and use instanceIndex as their Gaussian id.
struct RayHit
{
    float t{Ray::InfiniteDistance};
    float u{};
    float v{};
    uint32_t instanceIndex{InvalidIndex};
    uint32_t primitiveIndex{InvalidIndex};
};

struct CameraSample
{
    Ray ray{};
    float weight{1.0f};
};

struct PathRandomStreams
{
    RandomState opacity{};
    RandomState bsdf{};
    RandomState light{};
    RandomState shadow{};
    RandomState roulette{};
};

struct alignas(16) PathState
{
    SampledSpectrum throughput;
    SampledSpectrum radiance;
    SampledWavelengths wl;
    RandomState rngState;
    uint32_t depth;
    float alpha;
    float lastBsdfPdf;
    float etaScale;

    NR_CPU_GPU PathRandomStreams nextRandomStreams(
        const uint32_t pixel, const uint32_t accumulatedSamples)
    {
        const RandomState bounceKey = depth == 0
            ? seedRandom((static_cast<uint64_t>(accumulatedSamples) << 32u)
                | static_cast<uint64_t>(pixel))
            : rngState;
        rngState = advanceRandomSequence(bounceKey);
        return {
            forkRandom(bounceKey, RandomStream::Opacity),
            forkRandom(bounceKey, RandomStream::Bsdf),
            forkRandom(bounceKey, RandomStream::Light),
            forkRandom(bounceKey, RandomStream::Shadow),
            forkRandom(bounceKey, RandomStream::Roulette)};
    }

    NR_CPU_GPU void scatter(const SampledSpectrum& weight, const float pdf)
    {
        throughput *= weight;
        lastBsdfPdf = pdf;
        ++depth;
    }

    NR_CPU_GPU void transmit(const float eta)
    {
        etaScale *= eta * eta;
    }

};

static_assert(sizeof(PathState) == (NrSpectrumSamples == 1 ? 48 : 96));
