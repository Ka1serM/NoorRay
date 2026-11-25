#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/CameraRecords.h"
#include "NoorRay/Shaders/Common/Shared.h"
#include "NoorRay/Shaders/Materials/MaterialRecords.h"
#include "NoorRay/Shaders/Spectral/SpectralRecords.h"

NR_NAMESPACE_BEGIN

static const uint instanceMaskCamera = 1u;
static const uint instanceMaskShadow = 2u;
static const uint instanceMaskDiffuse = 4u;
static const uint instanceMaskGlossy = 8u;
static const uint instanceMaskTransmission = 16u;
static const uint instanceMaskVolumeScatter = 32u;

static const uint instanceFlagDoubleSided = 1u;
static const uint instanceFlagEmissive = 2u;
static const uint instanceFlagHoldout = 4u;

static const uint lightTypePoint = 0u;
static const uint lightTypeSpot = 1u;
static const uint lightTypeRect = 2u;
static const uint lightTypeDisk = 3u;
static const uint lightTypeSphere = 4u;
static const uint lightTypeDirectional = 5u;
static const uint lightTypeTriangle = 6u;

//! The bits of LightRecord.linking.z: a texture over the leaving direction, the falloff power, the light options, and from bit 8 the max bounces.
static const uint lightFlagLeavingTexture = 1u;
static const uint lightFalloffShift = 1u;
static const uint lightFalloffMask = 3u << lightFalloffShift;
static const uint lightFlagNoShadow = 1u << 3;
//! A triangle emitter whose front or back face light sampling skips, as Cycles' Emission Sampling.
static const uint lightFlagNoFront = 1u << 4;
static const uint lightFlagNoBack = 1u << 5;
//! The max bounces of a light, as Cycles: the vertices after this bounce get no light from it; 255 is no limit.
static const uint lightBouncesShift = 8u;
static const uint lightBouncesUnlimited = 255u;
//! The rays that do not see a light or the environment, as Cycles' Ray Visibility, in LightRecord.linking.z and in the scene flags.
static const uint lightHiddenDiffuse = 1u;
static const uint lightHiddenGlossy = 2u;
static const uint lightHiddenTransmission = 4u;
static const uint lightHiddenVolumeScatter = 8u;
static const uint lightHiddenShift = 16u;
static const uint lightHiddenMask = 15u;
static const uint lightFlagCamera = 1u << 20;

//! The bits of environmentParams.z: picks lights by power alone, camera rays see some lights, and from bit 2 the rays that do not see the environment.
static const uint sceneFlagByPower = 1u;
static const uint sceneFlagCameraLights = 2u;
static const uint sceneHiddenShift = 2u;

static const uint volumeKindHomogeneous = 0u;
static const uint volumeKindGrid = 1u;
//! A volume whose density graph the medium kernels run at each point; its sparse grid only gives the majorant.
static const uint volumeKindGraph = 2u;
//! A density graph baked to a sparse grid: cells of volumeBrickSize voxels, each a constant or a brick of 8-bit samples between the cell minimum and maximum.
static const uint volumeKindSparse = 3u;
static const uint volumeBrickSize = 8u;
//! A brick holds the samples at the voxel corners, so trilinear lookups stay inside one brick.
static const uint volumeBrickSamples = volumeBrickSize + 1u;
static const uint volumeBrickWords = (volumeBrickSamples * volumeBrickSamples * volumeBrickSamples + 3u) / 4u;

static const uint invalidIndex = 0xffffffffu;

//! The streams of a mesh with one value per triangle corner, at the index of the corner in indices; the other streams have one per vertex.
static const uint meshCornerNormals = 1u;
static const uint meshCornerTangents = 2u;
static const uint meshCornerUvs = 4u;
static const uint meshCornerColors = 8u;

//! Vertex streams in their packed encoding; positions holds three floats per vertex.
struct MeshRecord {
    //! boundsMin and boundsMax give the generated texture space.
    float4 boundsMin;
    float4 boundsMax;
    Ptr<float> positions;
    Ptr<uint> indices;
    Ptr<uint> normals;
    Ptr<uint> tangents;
    Ptr<uint> uv0;
    Ptr<uint> uv1;
    Ptr<uint> uv2;
    Ptr<uint> uv3;
    Ptr<uint> colors;
    //! The scene material index of each triangle.
    Ptr<uint> triangleMaterials;
    //! The positions and smooth vertex normals before the displacement, three floats each, or null.
    Ptr<float> restPositions;
    Ptr<float> restNormals;
    uint vertexCount;
    uint triangleCount;
    uint padding;
    uint cornerStreams;
};

//! Each transform is three rows of a 3x4 matrix.
struct InstanceData {
    float4 objectToWorld[3];
    float4 worldToObject[3];
    float4 previousObjectToWorld[3];
    //! Mesh, flags, override material or invalidIndex, first light; an emissive instance owns one triangle light per triangle.
    uint4 meta;
    //! The object colour as four halves in xy, the pass index and the random number of the object in 16 bits each, and the first of its attributes.
    uint4 info;
};

//! Radiance is rgb in W/(sr m^2) for area lights, W/sr for points and W/m^2 for directional lights.
struct LightRecord {
    //! xyz, and w the radius; a triangle keeps the radiance of its back face in position.w, tangent.w and bitangent.w.
    float4 position;
    //! The light's -Z axis, and w the cosine of the outer cone or of the directional half angle.
    float4 direction;
    //! tangent and bitangent are the half extents of a rect or the disk axes; tangent.w is the spot smoothstep width.
    float4 tangent;
    float4 bitangent;
    float4 radiance;
    //! Type, instance, primitive, unused.
    uint4 meta;
    //! xy the light set mask, as Cycles' light linking; z the light flags; w the falloff smooth.
    uint4 linking;
};

//! The light set of an instance in the high bits of its flags, and of a volume in the high bits of its kind; up to 64 sets.
static const uint lightSetShift = 24u;
static const uint lightSetCount = 64u;

//! A node of the light tree of a light set: the SG light of its cluster, as Tokuyoshi et al. 2024, "Hierarchical Light Sampling with Accurate SG Lighting".
struct LightTreeNode {
    //! The flux-weighted mean position of the lights, and w the spatial variance of their positions.
    float4 mean;
    //! The vMF axis of the emission, and w its sharpness; a distant light has its direction to the light and the sharpness of its disk.
    float4 axis;
    //! x is the intensity flux / (2 pi * the integral of the vMF), or the irradiance of a distant light; y is the bounding radius; z is the plane offset.
    float4 power;
    //! x is the second child of an interior node, whose first child follows it, or the light of a leaf; y holds the node flags.
    uint4 meta;
};

static const uint lightNodeLeaf = 1u;
//! The largest SG sharpness of a light, as Tokuyoshi's SGLIGHT_SHARPNESS_MAX, which keeps the SG lighting finite.
static const float lightSharpnessMax = 2199023255552.0f;
//! All lights of the node emit to the front of one plane: dot(axis, p) = power.z; a receiver behind it gets nothing.
static const uint lightNodePlanar = 2u;
//! The meta.x of the distant node of the environment.
static const uint lightNodeEnvironment = 0xfffffffeu;
//! A light set descends from depth 0 to at most 64, one bit of the trail of a light per depth.
static const uint lightTreeMaxDepth = 64u;

//! The first choice of a light set is among the nodes of a cut of its tree at this depth, which bound the receiver closer than the root.
static const uint lightTreeCutDepth = 3u;
static const uint lightTreeCutSize = 8u;
//! A cell of the visibility cache refines the cut of its light set up to this many nodes, as Wang et al. 2021, "Learning to Cluster for Rendering with Many Lights".
static const uint visibilityCutSize = 16u;
//! The first choices that the visibility cache learns: the environment, then the nodes of the cut of the cell, then the first distant lights.
static const uint visibilityChoices = 24u;
static const uint visibilityChoiceEnvironment = 0u;
static const uint visibilityChoiceCut = 1u;
static const uint visibilityChoiceDistant = visibilityChoiceCut + visibilityCutSize;
//! The words of a cell: the stats of its choices, then the nodes of its cut, then the light set that inserted it.
static const uint visibilityCellCut = visibilityChoices;
static const uint visibilityCellSet = visibilityCellCut + visibilityCutSize;
static const uint visibilityCellWords = visibilityCellSet + 1u;

//! The nodes of one light set: the tree from root, then the distant lights, whose importance does not fall off with distance.
struct LightTreeSet {
    uint root;
    uint nodeCount;
    uint firstDistant;
    uint distantCount;
    //! The nodes of the cut at lightTreeCutDepth from left to right, cutCount of them.
    uint cut[lightTreeCutSize];
    uint cutCount;
    uint3 padding;
};

//! Walker alias table: pick entry i uniformly, keep it when u < threshold, else take alias. pdf is the discrete probability of entry i.
struct AliasEntry {
    float threshold;
    uint alias;
    float pdf;
    uint unused;
};

//! The Gaussian TLAS holds one proxy instance per Gaussian with the cloud index as custom index.
struct GaussianCloudRecord {
    Ptr<float4> positionOpacity;
    Ptr<float4> rotations;
    Ptr<float4> scales;
    //! Half RGB triples, shStride halves per Gaussian.
    Ptr<uint> sh;
    float4 objectToWorld[3];
    float4 worldToObject[3];
    uint count;
    uint shDegree;
    uint shStride;
    uint firstInstance;
};

//! Coefficients are per metre in linear RGB.
struct VolumeRecord {
    float4 worldToObject[3];
    //! boundsMin.w is the density scale and boundsMax.w the phase anisotropy; the w of absorption, scattering and emission are the emission rate.
    float4 boundsMin;
    float4 boundsMax;
    float4 absorption;
    float4 scattering;
    float4 emission;
    uint4 resolution;
    //! Half pairs, x fastest.
    Ptr<uint> density;
    //! The bits of the largest grid density, which the device writes when it bakes the grid.
    AtomicPtr<uint> maximum;
    uint kind;
    //! The Draine alpha of the phase function; 0 is Henyey-Greenstein.
    float draineAlpha;
    //! The parameter values and textures of the density graph of a graph volume.
    Ptr<float4> graphParameters;
    Ptr<MaterialTextureRecord> graphTextures;
    //! Sparse: per cell x the half minimum and maximum, y the brick or invalidIndex for a constant cell; resolution counts the cells.
    Ptr<uint2> cells;
    Ptr<uint> bricks;
    //! Emission per metre as RGB9E5 at the voxels of resolution, and half pairs of the filled share of the mask cells; null for none.
    Ptr<uint> emissionGrid;
    Ptr<uint> mask;
    //! x is 1 with an emission grid.
    uint2 flags;
    //! w holds the bits of the smallest majorant, which gives an emissive volume enough collisions.
    uint4 maskResolution;
};

//! What camera rays that miss the scene see, in sceneCenter.w.
static const float environmentTransparent = 0.0f;
static const float environmentVisible = 1.0f;
static const float environmentBlack = 2.0f;

//! The root that every kernel reads.
struct SceneRecord {
    Ptr<MeshRecord> meshes;
    Ptr<InstanceData> instances;
    Ptr<MaterialRecord> materials;
    Ptr<LightRecord> lights;
    //! The light trees of all light sets, a LightTreeSet per set, and per set and light the bit trail from the root to the leaf of the light.
    Ptr<LightTreeNode> lightNodes;
    Ptr<LightTreeSet> lightSets;
    Ptr<uint2> lightTrails;
    Ptr<GaussianCloudRecord> gaussianClouds;
    Ptr<VolumeRecord> volumes;
    Ptr<AliasEntry> environmentAlias;
    Ptr<SpectralEntry> spectral;
    Ptr<ConductorSample> conductors;
    Ptr<uint> energyTables;
    AccelerationStructureHandle tlas;
    AccelerationStructureHandle gaussianTlas;
    //! The parameters of the graph materials, one float4 per slot.
    Ptr<float4> graphParameters;
    //! The triangle lights whose graph emission the material kernels bake each frame, grouped by graph program, and the first and count of each program.
    Ptr<uint> emitterLights;
    Ptr<uint2> emitterRanges;
    //! Mesh instances, lights, Gaussian clouds, volumes.
    uint4 counts;
    //! Texture, sampler, alias table width and height.
    uint4 environment;
    //! The texture and the sampler of the environment that camera rays see, for a world that shades them in another way; 0 for none.
    uint2 environmentCamera;
    //! The attributes of the instances, instanceAttributeCount for each.
    Ptr<float4> instanceAttributes;
    //! rgb and strength.
    float4 environmentColor;
    //! Rotation about +Y, the average environment luminance for the light tree (0 when not sampled), the scene flags as bits, scene radius.
    float4 environmentParams;
    //! xyz is the centre of the scene bounds; w is environmentTransparent, environmentVisible or environmentBlack.
    float4 sceneCenter;
};

NR_NAMESPACE_END

#if !defined(__SLANG__)
static_assert(sizeof(NoorRay::Shaders::MeshRecord) == 144);
static_assert(sizeof(NoorRay::Shaders::InstanceData) == 176);
static_assert(sizeof(NoorRay::Shaders::LightRecord) == 112);
static_assert(sizeof(NoorRay::Shaders::AliasEntry) == 16);
static_assert(sizeof(NoorRay::Shaders::GaussianCloudRecord) == 144);
static_assert(sizeof(NoorRay::Shaders::VolumeRecord) == 240);
static_assert(sizeof(NoorRay::Shaders::SceneRecord) == 240);
static_assert(sizeof(NoorRay::Shaders::LightTreeNode) == 64);
static_assert(sizeof(NoorRay::Shaders::LightTreeSet) == 64);
static_assert(offsetof(NoorRay::Shaders::SceneRecord, counts) == 144);
#endif
