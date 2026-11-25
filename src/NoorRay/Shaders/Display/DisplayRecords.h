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
    //! The upscaled image, read instead of source when set.
    TextureHandle image;
    uint padding;
};

//! Copies an image to a float4 buffer, so that a readback can read the upscaled image.
struct ReadImageArguments {
    TextureHandle image;
    uint width;
    uint height;
    Ptr<float4> destination;
    //! The film of the render size, which has the alpha.
    Ptr<float4> film;
    uint filmWidth;
    uint filmHeight;
};

//! Copies the upscaled image to a storage image that another API shares.
struct WriteImageArguments {
    TextureHandle image;
    uint width;
    uint height;
    StorageTextureHandle destination;
    Ptr<float4> film;
    uint filmWidth;
    uint filmHeight;
};

NR_NAMESPACE_END

#if !defined(__SLANG__)
#include <cstddef>
static_assert(sizeof(NoorRay::Shaders::DisplayArguments) == 40);
static_assert(offsetof(NoorRay::Shaders::DisplayArguments, width) == 8);
static_assert(offsetof(NoorRay::Shaders::DisplayArguments, toneMap) == 20);
static_assert(offsetof(NoorRay::Shaders::DisplayArguments, splats) == 24);
static_assert(sizeof(NoorRay::Shaders::ReadImageArguments) == 36);
static_assert(sizeof(NoorRay::Shaders::WriteImageArguments) == 32);
#endif
