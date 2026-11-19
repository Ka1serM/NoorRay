#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/CameraRecords.h"
#include "NoorRay/Shaders/Common/Shared.h"
#include "NoorRay/Shaders/Materials/MaterialRecords.h"
#include "NoorRay/Shaders/Spectral/SpectralRecords.h"

NR_NAMESPACE_BEGIN

static const uint instanceMaskCamera = 1u;
static const uint instanceMaskShadow = 2u;
static const uint instanceMaskIndirect = 4u;
static const uint instanceMaskSurfaces = 7u;

static const uint instanceFlagDoubleSided = 1u;
static const uint instanceFlagEmissive = 2u;

static const uint lightTypePoint = 0u;
static const uint lightTypeSpot = 1u;
static const uint lightTypeRect = 2u;
static const uint lightTypeDisk = 3u;
static const uint lightTypeSphere = 4u;
static const uint lightTypeDirectional = 5u;
static const uint lightTypeTriangle = 6u;

static const uint volumeKindHomogeneous = 0u;
static const uint volumeKindGrid = 1u;

static const uint invalidIndex = 0xffffffffu;

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
    uint vertexCount;
    uint triangleCount;
    uint uvSetCount;
    uint unused;
};

//! Each transform is three rows of a 3x4 matrix.
struct InstanceData {
    float4 objectToWorld[3];
    float4 worldToObject[3];
    float4 previousObjectToWorld[3];
    //! Mesh, flags, override material or invalidIndex, first light; an emissive instance owns one triangle light per triangle.
    uint4 meta;
};

//! Radiance is rgb in W/(sr m^2) for area lights, W/sr for points and W/m^2 for directional lights.
struct LightRecord {
    //! xyz, and w the radius.
    float4 position;
    //! The light's -Z axis, and w the cosine of the outer cone or of the directional half angle.
    float4 direction;
    //! tangent and bitangent are the half extents of a rect or the disk axes; tangent.w is the spot smoothstep width.
    float4 tangent;
    float4 bitangent;
    float4 radiance;
    //! Type, instance, primitive, unused.
    uint4 meta;
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
    //! boundsMin.w is the density scale and boundsMax.w the phase anisotropy.
    float4 boundsMin;
    float4 boundsMax;
    //! w is the largest grid density, 1 for homogeneous.
    float4 absorption;
    float4 scattering;
    float4 emission;
    uint4 resolution;
    //! Half pairs, x fastest.
    Ptr<uint> density;
    uint kind;
    uint unused;
};

//! The root that every kernel reads.
struct SceneRecord {
    Ptr<MeshRecord> meshes;
    Ptr<InstanceData> instances;
    Ptr<MaterialRecord> materials;
    Ptr<LightRecord> lights;
    Ptr<AliasEntry> lightAlias;
    Ptr<GaussianCloudRecord> gaussianClouds;
    Ptr<VolumeRecord> volumes;
    Ptr<AliasEntry> environmentAlias;
    Ptr<SpectralEntry> spectral;
    Ptr<ConductorSample> conductors;
    Ptr<uint> energyTables;
    AccelerationStructureHandle tlas;
    AccelerationStructureHandle gaussianTlas;
    uint2 unused;
    //! Mesh instances, lights, Gaussian clouds, volumes.
    uint4 counts;
    //! Texture, sampler, alias table width and height.
    uint4 environment;
    //! rgb and strength.
    float4 environmentColor;
    //! Rotation about +Y, environment selection probability, total light power, scene radius.
    float4 environmentParams;
    //! xyz is the centre of the scene bounds; w is 1 when camera rays that miss see the environment and 0 when they see black.
    float4 sceneCenter;
};

NR_NAMESPACE_END

#if !defined(__SLANG__)
static_assert(sizeof(NoorRay::Shaders::MeshRecord) == 128);
static_assert(sizeof(NoorRay::Shaders::InstanceData) == 160);
static_assert(sizeof(NoorRay::Shaders::LightRecord) == 96);
static_assert(sizeof(NoorRay::Shaders::AliasEntry) == 16);
static_assert(sizeof(NoorRay::Shaders::GaussianCloudRecord) == 144);
static_assert(sizeof(NoorRay::Shaders::VolumeRecord) == 160);
static_assert(sizeof(NoorRay::Shaders::SceneRecord) == 192);
static_assert(offsetof(NoorRay::Shaders::SceneRecord, counts) == 112);
#endif
