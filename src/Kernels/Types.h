#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#include <cuda.h>
#endif

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "GPU/Annotations.h"


static constexpr uint32_t InvalidIndex = ~0u;

struct alignas(16) PathState
{
    glm::vec3 throughput;
    glm::vec3 radiance;
    uint32_t rngState;
    uint32_t depth;
    uint32_t flags;
    uint32_t packedCounters;
    uint32_t lastBsdfPdfBits;
    uint32_t _pad0;
};
static_assert(sizeof(PathState) == 48);

struct alignas(16) PrimaryState
{
    glm::vec3 primaryAlbedo;
    glm::vec3 primaryNormal;
    glm::vec3 primaryPosition;
    uint32_t primaryObjectIndex;
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(PrimaryState) == 48);

struct alignas(16) PathRayWorkItem
{
    glm::vec3 origin;
    uint32_t sampleIndex;
    glm::vec3 direction;
    uint32_t _pad0;
};
static_assert(sizeof(PathRayWorkItem) == 32);

struct alignas(16) HitWorkItem
{
    glm::vec3 rayDirection;
    uint32_t sampleIndex;
    float baryU;
    float baryV;
    uint32_t instanceIndex;
    uint32_t primitiveIndex;
};
static_assert(sizeof(HitWorkItem) == 32);

struct alignas(16) ShadowWorkItem
{
    glm::vec3 origin;
    float tMin;
    glm::vec3 direction;
    float tMax;
    glm::vec3 contribution;
    uint32_t sampleIndex;
};
static_assert(sizeof(ShadowWorkItem) == 48);

struct WavefrontQueues
{
    uint32_t* rayCounts{};
    PathState* pathStates{};
    PrimaryState* primaryStates{};
    PathRayWorkItem* rayQueues[2]{};
    HitWorkItem* hitQueue{};
    ShadowWorkItem* shadowQueue{};
    uint32_t capacity{};
};
