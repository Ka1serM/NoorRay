#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/Shared.h"

NR_NAMESPACE_BEGIN

static const uint toneMapLinear = 0u;
static const uint toneMapNeutral = 1u;
static const uint toneMapAgx = 2u;

struct DisplayArguments {
    Ptr<float4> source;
    uint width;
    uint height;
    float exposure;
    uint toneMap;
    //! Premultiplied linear radiance of the sorted Gaussian splats, composited over the film when splatted is 1.
    TextureHandle splats;
    uint splatted;
};

NR_NAMESPACE_END

#if !defined(__SLANG__)
#include <cstddef>
static_assert(sizeof(NoorRay::Shaders::DisplayArguments) == 32);
static_assert(offsetof(NoorRay::Shaders::DisplayArguments, width) == 8);
static_assert(offsetof(NoorRay::Shaders::DisplayArguments, toneMap) == 20);
static_assert(offsetof(NoorRay::Shaders::DisplayArguments, splats) == 24);
#endif
