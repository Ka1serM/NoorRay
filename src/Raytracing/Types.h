#pragma once

#include <cstdint>

#if defined(NR_GPU_CODE)
#include <cuda.h>
#endif

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "CUDA/Annotations.h"
#include "Raytracing/Spectrum.h"
#include "Samplers/RandomSampler.h"


static constexpr uint32_t InvalidIndex = ~0u;

using TlasHandle = uint64_t;

// PathState now carries wavelengths + spectral throughput/radiance.
// lambda[]/lambdaPdf[] mirror SampledWavelengths fields and are stored
// inline to keep the struct self-contained without a separate queue buffer.
// Layout: 96 bytes (6 × 16-byte aligned lines).
struct alignas(16) PathState
{
    SampledSpectrum throughput;
    SampledSpectrum radiance;
    float lambda[NrSpectrumSamples];
    float lambdaPdf[NrSpectrumSamples];
    RandomState rngState;
    uint32_t depth;
    uint32_t flags;
    uint32_t packedCounters;
    uint32_t lastBsdfPdfBits;
    float etaScale;
};

struct alignas(16) PathRayWorkItem
{
    glm::vec3 origin;
    uint32_t sampleIndex;
    glm::vec3 direction;
};

struct alignas(16) HitWorkItem
{
    glm::vec3 rayDirection;
    uint32_t sampleIndex;
    float baryU;
    float baryV;
    uint32_t instanceIndex;
    uint32_t primitiveIndex;
};

struct alignas(16) ShadowWorkItem
{
    glm::vec3 origin;
    float tMin;
    glm::vec3 direction;
    float tMax;
    SampledSpectrum contribution;
    RandomState rngState;
    uint32_t sampleIndex;
};

struct WavefrontQueues
{
    uint32_t* rayCounts{};
    PathState* pathStates{};
    PathRayWorkItem* rayQueues[2]{};
    HitWorkItem* hitQueue{};
    ShadowWorkItem* shadowQueue{};
    PathRayWorkItem* aovRayQueue{};
    HitWorkItem* aovHitQueue{};
    uint32_t capacity{};
};
