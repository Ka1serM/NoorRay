#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/Shared.h"
#if !defined(__SLANG__)
#include "NoorRay/Shaders/Wavefront/WavefrontRecords.h"
#endif

NR_NAMESPACE_BEGIN

static const uint denoiseGroupSize = 8u;

//! The frame feeds RELAX: prepare writes its inputs and composite remodulates its outputs.
static const uint denoiseFlagRelax = 1u;
//! The frame feeds FSR: prepare writes its depth and motion and composite writes its colour.
static const uint denoiseFlagUpscale = 2u;
//! The frame feeds DLSS-RR: prepare writes its guides and composite writes its noisy colour.
static const uint denoiseFlagReconstruct = 4u;
//! RELAX also denoises the fog light, in a scene with volumes.
static const uint denoiseFlagFog = 8u;
//! Prepare and composite read the accumulated film and its split sums instead of this frame, for one denoise of a finished final render.
static const uint denoiseFlagFilm = 16u;

//! The guide planes of a film that denoises once: frames with a surface, then sums of diffuse albedo and roughness, specular albedo, normal and depth.
static const uint denoiseFilmSurfaces = 0u;
static const uint denoiseFilmDiffuseAlbedo = 1u;
static const uint denoiseFilmSpecularAlbedo = 5u;
static const uint denoiseFilmNormalDepth = 8u;
static const uint denoiseFilmStride = 12u;
//! The denoised film blends into the accumulated film with weight samples / (samples + this), the lowest error on the classroom at 64 and 512 samples.
static const float denoiseFilmBlend = 64.0f;

//! Diffuse and specular albedo divide the light by at least this, so dark materials do not blow up the denoiser input.
static const float denoiseMinAlbedo = 0.01f;
//! Blend clamps the accumulated mean to the 3x3 range of the denoised mean, widened by this share of the range, so lone bright samples do not show.
static const float denoiseBlendMargin = 0.25f;
//! A neighbour counts in the spherical harmonics resolve of a pixel when its depth differs by less than this share.
static const float denoiseShDepthTolerance = 0.05f;
//! The view depth of a pixel without a surface: beyond every denoising range.
static const float denoiseSkyViewZ = 1e7f;
//! The specular hit distance of DLSS-RR for a pixel without a surface: the largest half float.
static const float denoiseSkyHitDistance = 65504.0f;

//! The root arguments of the denoise kernels; prepare and gather run before Accumulate, composite after RELAX and blend after composite.
struct DenoiseArguments {
    Ptr<WavefrontFrame> frame;
    //! The camera of the previous frame, for the motion vectors.
    Ptr<CameraRecord> previousCamera;
    //! The guide sums that gather adds each frame to, denoiseFilmStride planes of one float per pixel.
    Ptr<float> filmSums;
    //! RELAX inputs: demodulated light and hit distance, normal and roughness, view depth, film motion in pixels and view depth change.
    StorageTextureHandle diffuse;
    StorageTextureHandle specular;
    StorageTextureHandle normalRoughness;
    StorageTextureHandle viewZ;
    StorageTextureHandle motion;
    //! The fog light in front of the primary vertex, which the RELAX diffuse denoiser of the fog filters without demodulation.
    StorageTextureHandle fog;
    //! The guides of the fog denoiser: a normal towards the camera, the view depth of the fog light, and the camera motion of that point.
    StorageTextureHandle fogNormalRoughness;
    StorageTextureHandle fogViewZ;
    StorageTextureHandle fogMotion;
    //! The first spherical harmonics band of the diffuse and specular light: the mean light direction times the demodulated luminance.
    StorageTextureHandle diffuseSh;
    StorageTextureHandle specularSh;
    //! FSR inputs: inverted depth and motion in render pixels from the current to the previous frame.
    StorageTextureHandle upscaleDepth;
    StorageTextureHandle upscaleMotion;
    //! The colour that FSR upscales or DLSS-RR reconstructs; composite writes the film output instead without upscaling.
    StorageTextureHandle color;
    //! DLSS-RR guides: albedos, world normal, linear roughness, linear view depth, motion in render pixels, and specular hit distance.
    StorageTextureHandle diffuseAlbedo;
    StorageTextureHandle specularAlbedo;
    StorageTextureHandle normals;
    StorageTextureHandle roughness;
    StorageTextureHandle depth;
    StorageTextureHandle reconstructMotion;
    StorageTextureHandle specularHitDistance;
    //! RELAX outputs: diffuse and specular light in YCoCg with their first harmonics band, and fog light.
    TextureHandle denoisedDiffuse;
    TextureHandle denoisedSpecular;
    TextureHandle denoisedDiffuseSh;
    TextureHandle denoisedSpecularSh;
    TextureHandle denoisedFog;
    //! The mean of the denoised frames of the accumulation, and their count in w.
    StorageTextureHandle denoisedMean;
    uint flags;
    //! The weight of the accumulated mean over the mean of the denoised frames is samples / (samples + this).
    float historyBlend;
};

//! The root arguments of the kernel that clears the lighting split of a film that denoises once, before the first frame of its accumulation.
struct DenoiseClearArguments {
    AtomicPtr<float> lighting;
    uint width;
    uint height;
};

NR_NAMESPACE_END

#if !defined(__SLANG__)
#include <cstddef>

static_assert(sizeof(NoorRay::Shaders::DenoiseArguments) == 140);
static_assert(offsetof(NoorRay::Shaders::DenoiseArguments, diffuseAlbedo) == 80);
static_assert(offsetof(NoorRay::Shaders::DenoiseArguments, flags) == 132);
#endif
