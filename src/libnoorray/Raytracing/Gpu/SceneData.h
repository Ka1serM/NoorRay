#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "CUDA/Unique/Texture.h"
#include "Raytracing/Gpu/Output.h"
#include "Shading/EnergyLut.h"
#include "Shading/SphericalHarmonics.h"
#include "Raytracing/Gpu/Types.h"
#include "Scene/GpuInstance.h"
#include "Scene/RenderSettings.h"
#include "Mesh/Assets/MeshAsset.h"
#include "Scene/Environment.h"
#include "Light/DirectionalLight.h"
#include "Light/PointLight.h"
#include "Light/SpotLight.h"
#include "Light/RectLight.h"

class Camera;
struct PsfGatherBucketSample;

// Host-built alias-table entry for analytic-light selection.  It is immutable
// scene data; PathIntegrator owns the selection policy.
struct AnalyticLightAliasEntry
{
    float threshold{};
    uint32_t alias{};
    float selectionPdf{};
};

struct GpuSceneData
{
    const MeshAsset* meshes{};
    const Material* materials{};
    const GpuInstance* instances{};
    const PointLight* pointLights{};
    const SpotLight* spotLights{};
    const RectLight* rectLights{};
    const DirectionalLight* directionalLights{};
    const AnalyticLightAliasEntry* analyticLightAliases{};
    const nr::cuda::UniqueTexture* textures{};
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
    uint32_t pointLightCount{};
    uint32_t spotLightCount{};
    uint32_t rectLightCount{};
    uint32_t directionalLightCount{};
    float analyticLightSelectionWeight{};
    uint32_t analyticLightAliasCount{};
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
    uint32_t aovQuery{};           // true only while the dedicated AOV raygen is launched
};

struct KernelParams
{
    GpuSceneData scene;
    OutputSurfaces output;
    OutputSurfaces alternateAovOutput;
    GpuFrameSettings frame;
    glm::vec4* accumulation{};
    glm::vec2* noiseMoments{}; // Per-pixel luminance mean and Welford M2.
    PsfGatherBucketSample* psfGatherBuckets{};
    uint32_t psfBinCount{};
    uint32_t depth{};
};
