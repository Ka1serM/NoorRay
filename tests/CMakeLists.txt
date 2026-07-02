add_subdirectory("${CMAKE_SOURCE_DIR}/external/Catch2" "${CMAKE_BINARY_DIR}/external/Catch2")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/Catch2/extras")
include(Catch)

function(add_noorray_test target source prefix)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(${target} PRIVATE Catch2::Catch2WithMain)
    catch_discover_tests(${target}
        TEST_PREFIX "${prefix}:"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endfunction()

add_noorray_test(sampler_sequence_test Sampler/SamplerSequence.cpp sampler)
add_noorray_test(sampler_distribution_test Sampler/SamplerDistribution.cpp sampler)

add_noorray_test(ggx_bsdf_test Bsdf/GgxBsdf.cpp bsdf)
target_link_libraries(ggx_bsdf_test PRIVATE glm CUDA::cudart)

add_noorray_test(dielectric_bsdf_test Bsdf/DielectricBsdf.cpp bsdf)
target_link_libraries(dielectric_bsdf_test PRIVATE glm CUDA::cudart)

add_noorray_test(bsdf_convention_test Bsdf/BsdfConvention.cpp bsdf)
target_link_libraries(bsdf_convention_test PRIVATE glm CUDA::cudart)

add_noorray_test(bsdf_cuda_test Bsdf/BsdfCuda.cu bsdf)
target_sources(bsdf_cuda_test PRIVATE "${CMAKE_SOURCE_DIR}/src/OpenPbr/OpenPbrLuts.cu")
target_link_libraries(bsdf_cuda_test PRIVATE glm CUDA::cudart)
set_target_properties(bsdf_cuda_test PROPERTIES CUDA_ARCHITECTURES ${NR_CUDA_ARCH})
target_compile_options(bsdf_cuda_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(openpbr_diagnostic Bsdf/OpenPbrDiagnostic.cu "${CMAKE_SOURCE_DIR}/src/OpenPbr/OpenPbrLuts.cu")
target_include_directories(openpbr_diagnostic PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(openpbr_diagnostic PRIVATE glm CUDA::cudart)
set_target_properties(openpbr_diagnostic PROPERTIES
    CUDA_ARCHITECTURES ${NR_CUDA_ARCH}
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(openpbr_diagnostic PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_noorray_test(spectrum_test Spectral/Spectrum.cpp spectral)
target_sources(spectrum_test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/Raytracing/RgbSpectrumTableData.cpp")
target_link_libraries(spectrum_test PRIVATE glm)

function(add_noorray_render_test target source)
    add_noorray_test(${target} ${source} e2e)
    target_compile_definitions(${target} PRIVATE
        TEST_SCENE_DIR="${CMAKE_SOURCE_DIR}/assets/tests"
        NOORRAY_EXECUTABLE="$<TARGET_FILE:NoorRay>"
    )
    target_link_libraries(${target} PRIVATE NoorRayIO)
    add_dependencies(${target} NoorRay)
endfunction()

add_noorray_render_test(furnace_test Bsdf/Furnace.cpp)
