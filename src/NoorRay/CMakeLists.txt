file(GLOB_RECURSE noorray_sources CONFIGURE_DEPENDS *.cpp)
if(PLATFORM_WEB)
    list(FILTER noorray_sources EXCLUDE REGEX "Desktop\\.cpp$")
else()
    list(FILTER noorray_sources EXCLUDE REGEX "Web\\.cpp$")
endif()
if(NOORRHI_BACKEND STREQUAL "WebGpu")
    list(FILTER noorray_sources EXCLUDE REGEX "Spirv\\.cpp$")
else()
    list(FILTER noorray_sources EXCLUDE REGEX "Wgsl\\.cpp$")
endif()
if(NOORRAY_USD_ROOT)
    list(FILTER noorray_sources EXCLUDE REGEX "/Import/UsdImporterDisabled\\.cpp$")
else()
    list(FILTER noorray_sources EXCLUDE REGEX "/Import/(UsdImporter|PreviewSurface)\\.cpp$")
endif()
if(NOORRAY_GAUSSIAN_TRACING)
    list(FILTER noorray_sources EXCLUDE REGEX "/Gpu/GaussianAccelerationStructureDisabled\\.cpp$")
else()
    list(FILTER noorray_sources EXCLUDE REGEX "/Gpu/GaussianAccelerationStructure\\.cpp$")
endif()

noorray_add_shaders(NoorRayShaders RAY_QUERIES
    KERNELS
        Advance
        BinPaths
        BuildSamples
        Generate
        Regenerate
        Finish
        VisibilityResolve
        Accumulate
        DenoisePrepare
        DenoiseGather
        DenoiseClear
        DenoiseComposite
        DenoiseBlend
        ToneMapVertex
        ToneMapFragment
        ReadImage
        WriteImage
        Extend:hasGaussians,hasVolumes,hasAlphaTest
        Shadow:hasGaussians,hasVolumes,hasAlphaTest,litGaussians
        ShadeMedium:hasGaussians,hasVolumes,litGaussians
        ShadeSurface:hasDispersion,hasAlphaTest,hasSubsurface
        ShadeSurfaceDirect:hasDispersion,hasAlphaTest,hasSubsurface
        ResolveShadowAlpha:hasAlphaTest
        ShadeRaster:hasAlphaTest
        EvaluateMaterials
        EvaluateFixedMaterials
        AlphaGraph
        VisibilityVertex
        VisibilityFragment
        ResetSplats
        ProjectSplats
        PrepareSplatSort
        CountSplatDigits
        ReduceSplatHistogram
        ScanSplatPartials
        DownsweepSplatHistogram
        ScatterSplatDigits
        SplatVertex
        SplatFragment
        RestirExtend:hasAlphaTest
        RestirBounce:hasAlphaTest,hasVolumes
        RestirResample:hasAlphaTest,hasVolumes
        RestirReplay:hasAlphaTest,hasVolumes
        RestirShade:hasAlphaTest,hasVolumes
        RestirDirect:hasAlphaTest,hasVolumes
        RestirFork
        RadianceCacheResolve
        DisplaceVertices
        DisplaceGraph
        DisplaceFaceNormals
        DisplaceVertexNormals
        DisplaceScatter
        BakeGraph
        AverageLightBake
    INCLUDES "${PROJECT_SOURCE_DIR}/external/RTXDI/Include" "${PROJECT_SOURCE_DIR}/external/SHARC/include")

#! RELAX diffuse and specular harmonics, RELAX diffuse radiance for the fog, and the float clear: the passes that nrd::Instance dispatches for NoorRay's settings.
set(relax_defines "NRD_SIGNAL=NRD_SIGNAL_BOTH,NRD_MODE=NRD_MODE_SH")
set(relax_radiance_defines "NRD_SIGNAL=NRD_SIGNAL_BOTH,NRD_MODE=NRD_MODE_RADIANCE")
set(fog_defines "NRD_SIGNAL=NRD_SIGNAL_DIFF,NRD_MODE=NRD_MODE_RADIANCE")
noorray_nrd_entries(NoorRayPostShaders nrd_entries PASSES
    NrdRelaxClassifyTiles:RELAX_ClassifyTiles
    NrdRelaxHitDistReconstruction3x3:RELAX_HitDistReconstruction:${relax_radiance_defines},MODE_5X5=0
    NrdRelaxHitDistReconstruction5x5:RELAX_HitDistReconstruction:${relax_radiance_defines},MODE_5X5=1
    NrdRelaxPrePass:RELAX_PrePass:${relax_defines}
    NrdRelaxTemporalAccumulation:RELAX_TemporalAccumulation:${relax_defines}
    NrdRelaxHistoryFix:RELAX_HistoryFix:${relax_defines}
    NrdRelaxHistoryClamping:RELAX_HistoryClamping:${relax_defines}
    NrdRelaxCopy:RELAX_Copy:${relax_defines}
    NrdRelaxAntiFirefly:RELAX_AntiFirefly:${relax_defines}
    NrdRelaxAtrousSmem:RELAX_AtrousSmem:${relax_defines}
    NrdRelaxAtrous:RELAX_Atrous:${relax_defines}
    NrdFogHitDistReconstruction3x3:RELAX_HitDistReconstruction:${fog_defines},MODE_5X5=0
    NrdFogHitDistReconstruction5x5:RELAX_HitDistReconstruction:${fog_defines},MODE_5X5=1
    NrdFogPrePass:RELAX_PrePass:${fog_defines}
    NrdFogTemporalAccumulation:RELAX_TemporalAccumulation:${fog_defines}
    NrdFogHistoryFix:RELAX_HistoryFix:${fog_defines}
    NrdFogHistoryClamping:RELAX_HistoryClamping:${fog_defines}
    NrdFogCopy:RELAX_Copy:${fog_defines}
    NrdFogAntiFirefly:RELAX_AntiFirefly:${fog_defines}
    NrdFogAtrousSmem:RELAX_AtrousSmem:${fog_defines}
    NrdFogAtrous:RELAX_Atrous:${fog_defines}
    NrdClear:Clear:FLOAT=1)
set(fsr_entries "")
foreach(pass Clear PrepareInputs LumaPyramid ShadingChangePyramid ShadingChange PrepareReactivity LumaInstability Accumulate AccumulateSharpen Rcas)
    list(APPEND fsr_entries "${CMAKE_CURRENT_SOURCE_DIR}/Shaders/Upscale/Fsr${pass}.slang")
endforeach()
#! NRD and FidelityFX shaders stay unmodified: NoorRay's wrappers include them with NoorRHI bindings, and Overrides shadows the SPD include for WGSL.
noorrhi_add_shaders(NoorRayPostShaders ${nrd_entries} ${fsr_entries}
    INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/Shaders" "${PROJECT_SOURCE_DIR}/src" "${NOORRAY_NRD_CONFIG_DIR}" "${PROJECT_SOURCE_DIR}/external/NRD/Shaders"
             "${PROJECT_SOURCE_DIR}/external/MathLib" "${CMAKE_CURRENT_SOURCE_DIR}/Shaders/Upscale/Overrides"
             "${PROJECT_SOURCE_DIR}/external/FidelityFX-SDK/sdk/include/FidelityFX/gpu" "${PROJECT_SOURCE_DIR}/external/FidelityFX-SDK/sdk/include")

add_library(NoorRay STATIC ${noorray_sources} $<TARGET_OBJECTS:NoorRayShaders> $<TARGET_OBJECTS:NoorRayPostShaders>)
add_library(NoorRay::NoorRay ALIAS NoorRay)
add_dependencies(NoorRay NoorRayShaders NoorRayPostShaders)
noorray_code_options(NoorRay)
target_compile_features(NoorRay PUBLIC cxx_std_23)
target_include_directories(NoorRay PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_include_directories(NoorRay PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/NoorRayShaders" "${CMAKE_CURRENT_BINARY_DIR}/NoorRayPostShaders")
target_link_libraries(NoorRay PUBLIC NoorRHI::NoorRHI rtxdi PRIVATE nrd fsr3upscaler stb tinygltf tinyobjloader tinyexr)
if(NOORRAY_USD_ROOT)
    target_link_libraries(NoorRay PRIVATE usd)
endif()
target_precompile_headers(NoorRay PRIVATE <algorithm> <cstdint> <expected> <memory> <span> <string> <string_view> <vector>)
