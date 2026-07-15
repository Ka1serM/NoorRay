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
static constexpr uint8_t MeshVisibility = uint8_t{1} << 0u;
static constexpr uint8_t GaussianVisibility = uint8_t{1} << 1u;
static constexpr uint8_t SceneVisibility = MeshVisibility | GaussianVisibility;
static constexpr uint8_t FullVisibility = ~uint8_t{0};

using TlasHandle = uint64_t;

// Bit offsets of the per-bounce-type counters packed into PathState::packedCounters.
static constexpr uint32_t CounterDiffuseShift = 0;
static constexpr uint32_t CounterSpecularShift = 8;
static constexpr uint32_t CounterTransmissionShift = 16;
static constexpr uint32_t CounterHitShift = 24;

// PathState carries the sampled wavelengths + spectral throughput/radiance,
// stored inline to keep the struct self-contained without a separate queue buffer.
struct alignas(16) PathState
{
    SampledSpectrum throughput;
    SampledSpectrum radiance;
    SampledWavelengths wl;
    RandomState rngState;
    uint32_t depth;
    uint32_t packedCounters;
    uint32_t lastBsdfPdfBits;
    float etaScale;
    glm::vec3 trainColor;
    float cameraWeight{1.0f};
};

struct alignas(16) PathRayWorkItem
{
    glm::vec3 origin;
    uint32_t sampleIndex;
    glm::vec3 direction;
};

// HitWorkItem carries per-hit data from Extend to Shade.  Fields are shared
// between mesh and gaussian hits through a union-like convention:
//   direction           — incident ray direction for every hit type
//   gaussianData        — inference: Gaussian hit position,
//                         training: incident ray origin
//   attribute0 — mesh: baryU, gaussian: unused
//   attribute1 — mesh: baryV, gaussian: unused
//   instanceIndex — mesh: instance id, gaussian: meshInstanceCount + gaussianId
//   primitiveIndex — mesh: triangle index, gaussian: unused
// The gaussian vs mesh discriminator is: instanceIndex >= meshInstanceCount.
struct alignas(16) HitWorkItem
{
    glm::vec3 direction;
    uint32_t sampleIndex;
    glm::vec3 gaussianData;
    float attribute0;
    float attribute1;
    uint32_t instanceIndex;
    uint32_t primitiveIndex;
};

static_assert(sizeof(HitWorkItem) == 48);

struct alignas(16) ShadowWorkItem
{
    glm::vec3 origin;
    float tMin;
    glm::vec3 direction;
    float tMax;
    SampledSpectrum contribution;
    RandomState rngState;
    uint32_t sampleIndex;
    uint32_t excludedGaussianId;
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
