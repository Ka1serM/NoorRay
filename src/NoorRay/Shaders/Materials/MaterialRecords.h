#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/Shared.h"

NR_NAMESPACE_BEGIN

//! The kinds of closure in closure.w of the inputs.
static const uint closureTranslucent = 1u;
static const uint closureHoldout = 2u;
static const uint closureRefractive = 3u;

//! A hit as the material sees it. normal.w is 1 when the ray hit the back of the surface; Materials::graphInputLayout reads the members by index.
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
    //! The hit in object space, and w a random number of the instance.
    float4 objectPosition;
    //! The hit in the mesh bounds from 0 to 1, and w a random number of the hit that picks a closure of a graph.
    float4 generated;
    //! The origin of the instance in world space.
    float4 objectLocation;
    //! dPdx and dPdy in object space, for the bump of a graph.
    float4 objectDx;
    float4 objectDy;
    //! The change of the generated coordinates per object space unit.
    float4 generatedScale;
    //! The rows of the world-to-object transform, which turn object space vectors as Cycles does.
    float4 worldToObject0;
    float4 worldToObject1;
    float4 worldToObject2;
    float4 objectToWorld0;
    float4 objectToWorld1;
    float4 objectToWorld2;
    //! The rows of the world-to-camera transform in Cycles' camera space: x right, y up, z forward.
    float4 worldToCamera0;
    float4 worldToCamera1;
    float4 worldToCamera2;
    //! The image position of a camera space point: xy scale and zw offset of x / z and y / z, or of x and y without perspective.
    float4 cameraWindow;
    //! The Light Path node: camera, shadow, diffuse and glossy ray.
    float4 rayType;
    //! The Light Path node: singular, reflection, transmission and volume scatter ray.
    float4 rayEvent;
    //! The Light Path node: ray depth, diffuse, glossy and transmission depth.
    float4 rayDepth;
    //! The colour of the object.
    float4 objectColor;
    //! The pass index of the object, the transparent depth, 1 for a perspective camera, and 0.
    float4 objectInfo;
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
    //! The weight of the closure over its probability, and w 1 when the diffuse lobe transmits.
    float4 closure;
    //! The interpolated normal before bump and normal maps, which Cycles calls sd->N.
    float4 smoothNormal;
    //! xyz is the subsurface radius in world units, w the share of the diffuse lobe that scatters below the surface.
    float4 subsurface;
};

//! BsdfInputs and the opacity of a hit as Materials/Closure.slang packs them: the material kernels write it, the shade kernels read it.
struct PackedClosure {
    uint4 base;
    uint4 frame;
    uint4 coat;
    uint4 sheen;
    uint4 film;
    uint4 subsurface;
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
    //! For a graph material: the graph, the first parameter in SceneRecord::graphParameters, and the alpha graph of an alpha-tested one.
    uint4 graph;
};

static const uint materialFlagAlphaTest = 1u;
static const uint materialFlagEmissive = 2u;
static const uint materialFlagDispersive = 4u;
static const uint materialFlagTransmissive = 8u;
static const uint materialFlagVertexColor = 16u;
static const uint materialFlagGlossiness = 32u;
static const uint materialFlagAlphaMask = 64u;
static const uint materialFlagGraph = 128u;
static const uint materialFlagSubsurface = 256u;
//! The shading starts from the normal before the displacement, which the bump of the whole displacement turns, as Cycles' Displacement and Bump.
static const uint materialFlagRestNormal = 512u;
//! The displacement bumps the normal of the surface itself, so Cycles' bump shadowing sees no bump.
static const uint materialFlagBumpsSurface = 1024u;

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
static_assert(sizeof(NoorRay::Shaders::ShadingContext) == 32 * 16);
static_assert(sizeof(NoorRay::Shaders::BsdfInputs) == 15 * 16);
static_assert(sizeof(NoorRay::Shaders::PackedClosure) == 96);
static_assert(sizeof(NoorRay::Shaders::MaterialTextureRecord) == 48);
static_assert(sizeof(NoorRay::Shaders::MaterialRecord) == 160);
static_assert(offsetof(NoorRay::Shaders::MaterialRecord, textures) == 128);
static_assert(offsetof(NoorRay::Shaders::MaterialRecord, flags) == 140);
#endif
