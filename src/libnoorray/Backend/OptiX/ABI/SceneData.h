#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <glm/vec3.hpp>

#include "Backend/CUDA/Unique/Texture.h"
#include "Materials/SVM/SvmTypes.h"
#include "Backend/OptiX/ABI/Output.h"
#include "Materials/Shading/EnergyLut.h"
#include "Materials/Shading/Material.h"
#include "Materials/Shading/SphericalHarmonics.h"
#include "Backend/OptiX/ABI/Types.h"
#include "Scene/GpuInstance.h"
#include "Scene/RenderSettings.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Scene/Resources/Environment.h"
#include "Rendering/Lighting/DirectionalLight.h"
#include "Rendering/Lighting/PointLight.h"
#include "Rendering/Lighting/SpotLight.h"
#include "Rendering/Lighting/RectLight.h"

class Camera;
struct PsfGatherBucketSample;

// Host-built alias-table entry for finite-light selection. It is immutable
// scene data; the path kernels own the selection policy. The table covers
// analytic lights and emissive mesh triangles in the same index space as
// directLightCandidates.
struct LightAliasEntry
{
    float threshold{};
    uint32_t alias{};
    float selectionPdf{};
};

// Flat, GPU-resident light-tree node. Interior nodes use the implicit left
// child (nodeIndex + 1) and store the right child explicitly. Keeping the
// layout pointer-free makes stochastic descent cheap in CUDA and lets the
// host build a balanced tree without mirroring any device-side objects.
struct LightTreeNode
{
    glm::vec4 sphere{}; // xyz = center, w = conservative radius
    float selectionWeight{};
    uint32_t childOrLightIndex{InvalidIndex};
    uint32_t parent{InvalidIndex};
    uint32_t flags{};
};
static_assert(sizeof(LightTreeNode) == 32);

inline constexpr uint32_t LightTreeLeaf = 1u << 0u;
inline constexpr uint32_t LightTreeHasDirectional = 1u << 1u;

enum class DirectLightType : uint32_t
{
    Point,
    Spot,
    Rect,
    Directional,
    MeshTriangle,
};

// Host-built light candidates. Analytic entries use `index` in their family
// array; mesh entries use instance/primitive to address the scene geometry.
// `selectionWeight` is the base proposal weight: mesh emitters are weighted by
// area and analytic lights are uniform when both classes are present. The
// light tree applies a conservative spatial bound to this weight at runtime.
struct DirectLightCandidate
{
    DirectLightType type{};
    uint32_t index{};
    uint32_t instanceIndex{InvalidIndex};
    uint32_t primitiveIndex{InvalidIndex};
    glm::vec3 position{};
    float area{};
    glm::vec3 normal{};
    float selectionWeight{}; // flat proposal weight (not emitted power)
    float powerBound{};       // conservative emitted-power estimate
    float spatialRadius{};    // bound around position for distance tests
    float orientationBound{}; // upper bound on emitter cosine
    glm::vec3 tangent{};      // rectangle local +U axis
    float width{};
    glm::vec3 bitangent{};    // cached rectangle local +V axis
    float height{};
    glm::vec3 color{1.0f};
    float intensity{1.0f};
    float innerCos{};         // spot-light inner cone cosine
    float outerCos{};         // spot-light outer cone cosine
    float invConeCosineRange{};
    float coneOneMinusCosine{}; // directional-light half-angle cache
    float coneProjectedArea{};
    float barnDoorLength{};
    float barnDoorExpansion{};
    uint32_t twoSided{};
    uint32_t barnDoorEnabled{};
    uint32_t lightTreeLeaf{InvalidIndex};
};

struct GpuSceneData
{
    const MeshAsset* meshes{};
    const Material* materials{};
    const GpuInstance* instances{};
    const LightAliasEntry* lightAliases{};
    const DirectLightCandidate* directLightCandidates{};
    const LightTreeNode* lightTreeNodes{};
    // Primitive-to-candidate mappings for the separate analytic and mesh-light
    // GAS instances. Mesh-light visibility is opt-in for path rays because
    // emissive triangles are already present in the regular mesh TLAS.
    const uint32_t* analyticLightBvhCandidateIndices{};
    const uint32_t* meshLightBvhCandidateIndices{};
    const uint32_t* meshLightCandidateOffsets{};
    const uint32_t* meshLightCandidateIndices{};
    const nr::cuda::UniqueTexture* textures{};
    const std::uint32_t* svmWords{};
    const std::uint32_t* svmTextureIndices{};
    TlasHandle tlasHandle{};
    RenderSettings renderSettings{};
    const Environment* environment{};
    const Camera* camera{};
    // Jakob & Hanika sRGB→spectrum table (64^3, 9 MB on device).
    const float* spectrumTableScale{};   // 64 floats (z-nodes, non-uniform)
    const float* spectrumTableCoeffs{};  // float[3][64][64][64][3]
    cudaTextureObject_t spectrumTableTexture{}; // hardware-filtered float4 coefficient LUT
    nr::shading::energy_lut::Textures energyLuts{};
    const float* d65{};                  // CIE D65, 300--830 nm at 5 nm
    // CIE 1931 2-degree CMFs (471 floats each, device pointers).
    const float* cieX{};
    const float* cieY{};
    const float* cieZ{};
    uint32_t directionalLightCount{};
    uint32_t directionalLightCandidateOffset{};
    float lightSelectionWeight{};
    // Cached finite/environment mixture probabilities. These are constant for
    // the scene upload and are queried from several hot MIS paths per bounce.
    float finiteLightProbability{};
    float environmentLightProbability{};
    // Conservative scene-wide fast-path flag. It is true only when the SVM
    // bytecode contains no opacity-producing surface nodes.
    uint32_t allMaterialsOpaque{1};
    uint32_t lightAliasCount{};
    uint32_t lightTreeNodeCount{};
    uint32_t directLightCandidateCount{};
    uint32_t meshLightCandidateIndexCount{};
    uint32_t meshLightInstanceCount{};
    uint32_t analyticLightBvhPrimitiveCount{};
    uint32_t meshLightBvhPrimitiveCount{};
    uint32_t textureCount{};
    // Gaussian splat data — flat GPU arrays.
    const float* gaussianOpacities{}; // sigmoid opacity, indexed by global id
    const __half* gaussianShCoeffs{};
    uint32_t gaussianShCoefficientCount{MaxSphericalHarmonicsCoefficientCount};
    const uint32_t* gaussianInstanceOffsets{};
    uint32_t gaussianCount{};
    uint32_t meshInstanceCount{}; // number of mesh instance slots (id space split)

};

struct GpuFrameSettings
{
    uint32_t width{};
    uint32_t height{};
    uint32_t totalAccumulated{};   // total samples accumulated so far (blend weight)
    uint32_t visibilityMask{SceneVisibility};
    float cutoffDistanceSq{};      // cutoff sigma squared, precomputed on host
    uint32_t frameIndex{};         // 0 while accumulation is being reset (for example, camera motion)
    uint32_t aovQuery{};           // true while the first path raygen writes AOVs
    uint32_t serEnabled{1};        // beauty SER is enabled by default; debug-toggleable
    uint32_t writeOutput{1};       // only the final queued spp needs a display-surface write
};

struct KernelParams
{
    GpuSceneData scene;
    OutputSurfaces output;
    GpuFrameSettings frame;
    glm::vec4* accumulation{};
    // Optional pitch-linear OptiX guide buffers. The path tracer writes these
    // alongside the display/export AOV images when an AOV refresh runs.
    float3* denoiserAlbedoGuide{};
    float3* denoiserNormalGuide{};
    PsfGatherBucketSample* psfGatherBuckets{};
    uint32_t psfBinCount{};
    uint32_t depth{};
};
