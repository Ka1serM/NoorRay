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

set(schema_output "${CMAKE_CURRENT_BINARY_DIR}/Generated/NoorRay/Schema")
set(schema_headers "")
foreach(schema NoorRayScene)
    set(schema_source "${PROJECT_SOURCE_DIR}/schema/${schema}.fbs")
    set(schema_header "${schema_output}/${schema}_generated.h")
    add_custom_command(OUTPUT "${schema_header}"
                       COMMAND "${CMAKE_COMMAND}" -E make_directory "${schema_output}"
                       COMMAND ${FLATC_COMMAND} --cpp --scoped-enums --no-emit-min-max-enum-values -I "${PROJECT_SOURCE_DIR}/schema" -o "${schema_output}"
                               "${schema_source}"
                       DEPENDS "${schema_source}" ${FLATC_DEPENDENCY}
                       VERBATIM)
    list(APPEND schema_headers "${schema_header}")
endforeach()

noorray_add_shaders(NoorRayShaders RAY_QUERIES
    KERNELS
        Advance
        BinPaths
        BuildSamples
        Generate
        Regenerate
        Finish
        Accumulate
        ToneMapVertex
        ToneMapFragment
        Extend:hasGaussians,hasVolumes,hasAlphaTest
        Shadow:hasGaussians,hasVolumes,hasAlphaTest,litGaussians
        ShadeMedium:hasGaussians,hasVolumes,litGaussians
        ShadeSurface:hasDispersion,hasAlphaTest
        ResolveShadowAlpha:hasAlphaTest
        ShadeRaster:hasAlphaTest
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
        RestirPrimary:hasAlphaTest
        RestirInitial:hasAlphaTest
        RestirTemporal:hasAlphaTest
        RestirSpatial:hasAlphaTest
        RestirShade:hasAlphaTest
    INCLUDES "${PROJECT_SOURCE_DIR}/external/RTXDI/Include")

add_library(NoorRay STATIC ${noorray_sources} ${schema_headers} $<TARGET_OBJECTS:NoorRayShaders>)
add_library(NoorRay::NoorRay ALIAS NoorRay)
add_dependencies(NoorRay NoorRayShaders)
noorray_code_options(NoorRay)
target_compile_features(NoorRay PUBLIC cxx_std_23)
target_include_directories(NoorRay PUBLIC "${PROJECT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/Generated")
target_include_directories(NoorRay PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/NoorRayShaders")
target_link_libraries(NoorRay PUBLIC NoorRHI::NoorRHI FlatBuffers::FlatBuffers rtxdi PRIVATE stb tinygltf tinyobjloader tinyexr)
if(NOORRAY_USD_ROOT)
    target_link_libraries(NoorRay PRIVATE usd)
endif()
target_precompile_headers(NoorRay PRIVATE <algorithm> <cstdint> <expected> <memory> <span> <string> <string_view> <vector>)
