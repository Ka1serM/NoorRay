#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/Shared.h"

NR_NAMESPACE_BEGIN

//! A hit as the material sees it. normal.w is 1 when the ray hit the back of the surface.
struct ShadingContext {
    float4 position;
    float4 normal;
    float4 geometricNormal;
    float4 tangent;
    float4 incoming;
    float4 uv01;
    float4 uv23;
    float4 color;
    float4 uvGradients;
    float4 dPdx;
    float4 dPdy;
};

struct BsdfInputs {
    float4 baseColorAlpha;
    float4 metallicRoughnessIorSpecular;
    float4 specularTintAnisotropic;
    float4 normalRotation;
    float4 tangentTransmission;
    float4 coat;
    float4 coatTint;
    float4 coatNormal;
    float4 sheen;
    float4 sheenTint;
    float4 emission;
    float4 thinFilmDispersion;
};

//! One texture of a material slot: texture and sampler handle, UV set, channel of a scalar slot, and the UV transform rows (xyz).
struct MaterialTextureRecord {
    uint texture;
    uint sampler;
    uint uvSet;
    uint channel;
    float4 row0;
    float4 row1;
};

//! The fixed Principled material. textures holds the records of the set slots in slot order; bit n of textureMask is set when slot n has a texture.
struct MaterialRecord {
    float4 baseColorAlpha;
    float4 metallicRoughnessIorSpecular;
    float4 specularTintAnisotropic;
    //! Transmission, anisotropic rotation in turns, coat, coat roughness.
    float4 transmissionRotationCoat;
    float4 sheenColorRoughness;
    //! Emission colour and strength.
    float4 emission;
    //! Thin film thickness and IOR, dispersion, alpha cutoff.
    float4 thinFilmDispersion;
    //! Normal map scale, coat normal map scale.
    float4 normalScales;
    Ptr<MaterialTextureRecord> textures;
    uint textureMask;
    uint flags;
};

static const uint materialFlagAlphaTest = 1u;
static const uint materialFlagEmissive = 2u;
static const uint materialFlagDispersive = 4u;
static const uint materialFlagTransmissive = 8u;
static const uint materialFlagVertexColor = 16u;
static const uint materialFlagGlossiness = 32u;
static const uint materialFlagAlphaMask = 64u;

static const uint textureSlotBaseColor = 0;
static const uint textureSlotAlpha = 1;
static const uint textureSlotMetallic = 2;
static const uint textureSlotRoughness = 3;
static const uint textureSlotNormal = 4;
static const uint textureSlotSpecular = 5;
static const uint textureSlotTransmission = 6;
static const uint textureSlotCoat = 7;
static const uint textureSlotCoatRoughness = 8;
static const uint textureSlotCoatNormal = 9;
static const uint textureSlotEmission = 10;

NR_NAMESPACE_END

#if !defined(__SLANG__)
static_assert(sizeof(NoorRay::Shaders::ShadingContext) == 11 * 16);
static_assert(sizeof(NoorRay::Shaders::BsdfInputs) == 12 * 16);
static_assert(sizeof(NoorRay::Shaders::MaterialTextureRecord) == 48);
static_assert(sizeof(NoorRay::Shaders::MaterialRecord) == 144);
static_assert(offsetof(NoorRay::Shaders::MaterialRecord, textures) == 128);
static_assert(offsetof(NoorRay::Shaders::MaterialRecord, flags) == 140);
#endif
