#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "Camera/Camera.h"
#include "CUDA/Unique/Texture.h"
#include "Raytracing/Output.h"
#include "Raytracing/Types.h"
#include "Scene/GpuInstance.h"
#include "Scene/RenderSettings.h"
#include "Mesh/MeshAsset.h"
#include "Mesh/OpenPbrEnergy.h"
#include "Scene/Environment.h"
#include "Light/DirectionalLight.h"
#include "Light/PointLight.h"
#include "Light/SpotLight.h"
#include "Light/RectLight.h"

struct GpuSceneData
{
    const MeshAsset* meshes{};
    const GpuInstance* instances{};
    const PointLight* pointLights{};
    const SpotLight* spotLights{};
    const RectLight* rectLights{};
    const DirectionalLight* directionalLights{};
    const nr::cuda::UniqueTexture* textures{};
    TlasHandle tlasHandle{};
    RenderSettings renderSettings{};
    const Environment* environment{};
    const Camera* camera{};
    // Jakob & Hanika sRGB→spectrum table (64^3, 9 MB on device).
    const float* spectrumTableScale{};   // 64 floats (z-nodes, non-uniform)
    const float* spectrumTableCoeffs{};  // float[3][64][64][64][3]
    const float* d65{};                  // CIE D65, 300--830 nm at 5 nm
    // CIE 1931 2-degree CMFs (471 floats each, device pointers).
    const float* cieX{};
    const float* cieY{};
    const float* cieZ{};
    // OpenPBR opaque-dielectric energy-compensation LUTs (hardware-filtered).
    nr::openpbr::EnergyLutTextures openPbrLuts{};
    uint32_t pointLightCount{};
    uint32_t spotLightCount{};
    uint32_t rectLightCount{};
    uint32_t directionalLightCount{};
    float analyticLightSelectionWeight{};
    uint32_t textureCount{};
    // Gaussian splat data — flat GPU arrays.
    const float* gaussianOpacities{}; // sigmoid opacity, indexed by global id
    const glm::vec3* gaussianSpectrumCoeffs{}; // precomputed (c0,c1,c2), indexed by global id
    uint32_t gaussianCount{};
    uint32_t meshInstanceCount{}; // number of mesh instance slots (id space split)
};

struct GpuFrameSettings
{
    uint32_t width{};
    uint32_t height{};
    uint32_t totalAccumulated{};   // total samples accumulated so far (blend weight)
    uint32_t visibilityMask{0x03}; // 0x01 = mesh, 0x02 = gaussian, 0x03 = both
    float cutoffDistanceSq{};      // cutoff sigma squared, precomputed on host
    int frameIndex{};              // pushData.frame; 0 while resetAccumulation keeps retriggering (camera in motion)
};

struct KernelParams
{
    GpuSceneData scene;
    WavefrontQueues queues;
    OutputSurfaces output;
    GpuFrameSettings frame;
    glm::vec4* accumulation{};
    PsfGatherBucketSample* psfGatherBuckets{};
    uint32_t psfBinCount{};
    uint32_t depth{};
};
