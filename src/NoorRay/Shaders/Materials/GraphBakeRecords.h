#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/Shared.h"
//! Slang has the material records from the SceneRecords module, which the bake kernel imports.
#if !defined(__SLANG__)
#include "NoorRay/Shaders/Materials/MaterialRecords.h"
#endif

NR_NAMESPACE_BEGIN

static const uint graphBakeGroupSize = 64u;
//! The groups of one dispatch row; a grid of more groups than a dispatch side allows takes more rows.
static const uint graphBakeRowGroups = 4096u;
static const uint lightAverageGroupSize = 256u;

//! Bake kinds: a spot gobo over its UV, a rect or disk over its UV square or leaving direction, a point or sphere over map_to_sphere, two volume passes.
static const uint graphBakeSpot = 0u;
static const uint graphBakeArea = 1u;
static const uint graphBakeSphere = 2u;
//! Writes the range of each cell, counts the cells that need a brick, and raises the largest density.
static const uint graphBakeVolumeCells = 3u;
//! Gives each cell that needs a brick a slot and fills it.
static const uint graphBakeVolumeBricks = 4u;
static const uint graphBakeAreaDirection = 5u;
//! The world over the equirect map of the environment, as a diffuse ray and as a camera ray sees it.
static const uint graphBakeWorld = 6u;
static const uint graphBakeWorldCamera = 7u;

//! One bake of a light or volume graph: a light texel is the RGB9E5 emission at the texel centre; a volume samples emission red at the voxel corners.
struct GraphBakeRecord {
    Ptr<float4> parameters;
    Ptr<MaterialTextureRecord> textures;
    //! Light: the texels. Volume: the cells, two words each.
    Ptr<uint> output;
    uint kind;
    uint unused;
    //! The rows of the object to world transform.
    float4 objectToWorld[3];
    //! Light: x and y the local half sizes of an area light, z the radius, w the spot factor 0.5 / tan(half angle). Volume: the local bounds minimum.
    float4 shape;
    //! Volume: the local bounds maximum.
    float4 boundsMax;
    //! Volume: xyz the cells and w the samples of a brick.
    uint4 resolution;
    //! Volume: the bits of the largest density, then the count of bricks, then the next free brick slot.
    AtomicPtr<uint> maximum;
    Ptr<uint> bricks;
};

struct GraphBakeArguments {
    Ptr<GraphBakeRecord> record;
};

//! The mean of a light texture; a spot mean covers only the texels inside its cone.
struct LightAverageArguments {
    Ptr<uint> texels;
    Ptr<float4> average;
    uint resolution;
    uint kind;
};

NR_NAMESPACE_END

#if !defined(__SLANG__)
static_assert(sizeof(NoorRay::Shaders::GraphBakeRecord) == 32 + 48 + 48 + 16);
#endif
