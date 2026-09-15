# The gpu test suite. Catch2 and slangc are test-only dependencies; the
# library itself needs neither. SDL3 is optional and gates the presentation
# test, so the suite still builds on a machine without it.
find_program(GPU_SLANGC_EXECUTABLE NAMES slangc
    HINTS "$ENV{VULKAN_SDK}/bin"
    REQUIRED)

if(NOT TARGET Catch2::Catch2WithMain)
    find_package(Catch2 3 QUIET)
endif()
if(NOT TARGET Catch2::Catch2WithMain)
    # Vendored location once this is its own repository, then the copy in the
    # surrounding NoorRay tree so a standalone configure works before the split.
    foreach(_catch2
            "${CMAKE_CURRENT_SOURCE_DIR}/../external/Catch2"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../external/Catch2")
        if(EXISTS "${_catch2}/CMakeLists.txt")
            add_subdirectory("${_catch2}" "${CMAKE_BINARY_DIR}/external/Catch2")
            break()
        endif()
    endforeach()
endif()
if(NOT TARGET Catch2::Catch2WithMain)
    message(FATAL_ERROR
        "Catch2 v3 not found. Install it or configure with -DGPU_BUILD_TESTS=OFF.")
endif()
list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
include(Catch OPTIONAL RESULT_VARIABLE GPU_CATCH_MODULE)

set(GPU_TEST_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(GPU_ADD_SHADER "${GPU_TEST_SHADER_DIR}/add.spv")
set(GPU_TRIANGLE_SHADER "${GPU_TEST_SHADER_DIR}/triangle.spv")
set(GPU_RAYTRACING_SHADER "${GPU_TEST_SHADER_DIR}/raytracing.spv")
set(GPU_IMAGE_SHADER "${GPU_TEST_SHADER_DIR}/image.spv")

# Most tests use BDA buffers and framebuffer attachments, not shader textures,
# so only image.slang requests the descriptor-heap capability.
set(GPU_TEST_SHADER_FLAGS
    -emit-spirv-directly
    -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081)

add_custom_command(
    OUTPUT "${GPU_ADD_SHADER}" "${GPU_TRIANGLE_SHADER}" "${GPU_RAYTRACING_SHADER}"
        "${GPU_IMAGE_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${GPU_TEST_SHADER_DIR}"
    COMMAND "${GPU_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/image.slang"
        -entry main -stage compute -target spirv
        -profile spirv_1_5 -capability spvDescriptorHeapEXT ${GPU_TEST_SHADER_FLAGS}
        -o "${GPU_IMAGE_SHADER}"
    COMMAND "${GPU_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/add.slang"
        -entry main -stage compute -target spirv
        -profile spirv_1_5 ${GPU_TEST_SHADER_FLAGS}
        -o "${GPU_ADD_SHADER}"
    COMMAND "${GPU_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.slang"
        -entry vertMain -stage vertex -entry fragMain -stage fragment
        -target spirv -profile spirv_1_5 ${GPU_TEST_SHADER_FLAGS}
        -o "${GPU_TRIANGLE_SHADER}"
    COMMAND "${GPU_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/raytracing.slang"
        -entry rayGenMain -stage raygeneration
        -entry missMain -stage miss
        -entry closestHitMain -stage closesthit
        -target spirv -profile spirv_1_5+spvRayTracingKHR ${GPU_TEST_SHADER_FLAGS}
        -o "${GPU_RAYTRACING_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/add.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/raytracing.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/image.slang"
    COMMENT "Compiling gpu test Slang shaders" VERBATIM)

add_custom_target(gpu_test_shaders DEPENDS
    "${GPU_ADD_SHADER}" "${GPU_TRIANGLE_SHADER}" "${GPU_RAYTRACING_SHADER}"
    "${GPU_IMAGE_SHADER}")

function(add_gpu_test target source)
    add_executable(${target} "${source}")
    add_dependencies(${target} gpu_test_shaders)
    target_compile_features(${target} PRIVATE cxx_std_23)
    target_link_libraries(${target} PRIVATE gpu::gpu Catch2::Catch2WithMain)
    target_compile_definitions(${target} PRIVATE ${ARGN})
    if(COMMAND catch_discover_tests)
        catch_discover_tests(${target} TEST_PREFIX "gpu:"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" PROPERTIES LABELS "gpu")
    else()
        add_test(NAME "gpu:${target}" COMMAND ${target})
    endif()
endfunction()

add_gpu_test(gpu_compute_test GpuComputeTest.cpp
    "GPU_TEST_SHADER=\"${GPU_ADD_SHADER}\""
    "GPU_TRIANGLE_SHADER=\"${GPU_TRIANGLE_SHADER}\""
    "GPU_IMAGE_SHADER=\"${GPU_IMAGE_SHADER}\"")

add_gpu_test(gpu_ray_tracing_test GpuRayTracingTest.cpp
    "GPU_RAYTRACING_SHADER=\"${GPU_RAYTRACING_SHADER}\"")

# Presentation needs a real window system. Skipped entirely when no SDL3 is
# available, rather than forcing a windowing dependency on the whole suite.
if(NOT TARGET SDL3::SDL3-static AND NOT TARGET SDL3::SDL3)
    find_package(SDL3 QUIET)
endif()
if(TARGET SDL3::SDL3-static OR TARGET SDL3::SDL3)
    add_gpu_test(gpu_presentation_test GpuPresentationTest.cpp)
    if(TARGET SDL3::SDL3-static)
        target_link_libraries(gpu_presentation_test PRIVATE SDL3::SDL3-static)
    else()
        target_link_libraries(gpu_presentation_test PRIVATE SDL3::SDL3)
    endif()
else()
    message(STATUS "gpu: SDL3 not found, skipping gpu_presentation_test")
endif()
