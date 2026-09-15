#pragma once

#include "Types.h"

#ifdef __cplusplus
#include <cstddef>
namespace nr::graphics {
#endif

struct Environment
{
    float3 color;
    float rotationSin;
    float rotationCos;
    float visibleExposureScale;
    float lightingExposureScale;
    float importanceWeight;
    // Resource-descriptor-heap indices; 0 means no texture. The padding keeps
    // cdfWidth and everything after it at their former offsets.
    uint texture;
    uint cdfTexture;
    uint texturePadding0;
    uint texturePadding1;
    int cdfWidth;
    int cdfHeight;
    int mapping;
    int padding0;
    float4 environmentFromWorld[3];
};

#ifdef __cplusplus
inline constexpr std::uint32_t EnvironmentNoTexture = 0;

static_assert(offsetof(Environment, texture) == 32);
static_assert(offsetof(Environment, cdfWidth) == 48);
static_assert(offsetof(Environment, environmentFromWorld) == 64);
static_assert(sizeof(Environment) == 112);

} // namespace nr::graphics
#endif
