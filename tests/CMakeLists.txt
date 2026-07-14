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

add_noorray_test(sampler_sequence_test SamplerSequenceTest.cpp sampler)
add_noorray_test(sampler_distribution_test SamplerDistributionTest.cpp sampler)

add_noorray_test(pbrt_parser_test
    "PbrtParserTest.cpp;${CMAKE_SOURCE_DIR}/src/libnoorray/Scene/PbrtParser.cpp" pbrt)
target_include_directories(pbrt_parser_test PRIVATE "${CMAKE_SOURCE_DIR}/src/libnoorray")

add_noorray_test(opaque_bsdf_test OpaqueBsdfTest.cpp bsdf)
target_link_libraries(opaque_bsdf_test PRIVATE glm CUDA::cudart)

add_noorray_test(ggx_bsdf_test GgxBsdfTest.cpp bsdf)
target_link_libraries(ggx_bsdf_test PRIVATE glm CUDA::cudart)

add_noorray_test(dielectric_bsdf_test DielectricBsdfTest.cpp bsdf)
target_link_libraries(dielectric_bsdf_test PRIVATE glm CUDA::cudart)

function(add_noorray_render_test target source)
    add_noorray_test(${target} ${source} e2e)
    target_compile_definitions(${target} PRIVATE
        TEST_SCENE_DIR="${CMAKE_SOURCE_DIR}/assets/tests"
        NOORRAY_EXECUTABLE="$<TARGET_FILE:NoorRay>"
    )
    target_link_libraries(${target} PRIVATE NoorRayIO)
    add_dependencies(${target} NoorRay)
endfunction()

add_noorray_render_test(render_smoke_test RenderSmokeTest.cpp)
add_noorray_render_test(white_furnace_test WhiteFurnaceTest.cpp)
