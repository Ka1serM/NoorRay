# The examples are the gpu library's own demonstration programs. They need
# slangc to produce SPIR-V; the library itself does not.
find_program(GPU_SLANGC_EXECUTABLE NAMES slangc
    HINTS "$ENV{VULKAN_SDK}/bin"
    REQUIRED)

set(GPU_EXAMPLE_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(GPU_EXAMPLE_COMPUTE_SHADER "${GPU_EXAMPLE_SHADER_DIR}/compute.spv")
set(GPU_EXAMPLE_GRAPHICS_SHADER "${GPU_EXAMPLE_SHADER_DIR}/triangle.spv")
set(GPU_EXAMPLE_RAYGEN_SHADER "${GPU_EXAMPLE_SHADER_DIR}/raygen.spv")

# Compiled for the descriptor-heap model the library's texture handles use.
# The resource heap stride stays at Slang's default (the image descriptor
# size), which is how the device lays out its slots.
set(GPU_EXAMPLE_SHADER_FLAGS
    -capability spvDescriptorHeapEXT -emit-spirv-directly
    -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081)

add_custom_command(
    OUTPUT "${GPU_EXAMPLE_COMPUTE_SHADER}"
           "${GPU_EXAMPLE_GRAPHICS_SHADER}"
           "${GPU_EXAMPLE_RAYGEN_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${GPU_EXAMPLE_SHADER_DIR}"
    COMMAND "${GPU_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute.slang"
        -entry main -stage compute -target spirv
        -profile spirv_1_5+SPV_EXT_descriptor_heap
        ${GPU_EXAMPLE_SHADER_FLAGS}
        -o "${GPU_EXAMPLE_COMPUTE_SHADER}"
    COMMAND "${GPU_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.slang"
        -entry vertMain -stage vertex -entry fragMain -stage fragment
        -target spirv -profile spirv_1_5+SPV_EXT_descriptor_heap
        ${GPU_EXAMPLE_SHADER_FLAGS}
        -o "${GPU_EXAMPLE_GRAPHICS_SHADER}"
    COMMAND "${GPU_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/raygen.slang"
        -entry main -stage raygeneration -target spirv
        -profile spirv_1_5+spvRayTracingKHR+SPV_EXT_descriptor_heap
        ${GPU_EXAMPLE_SHADER_FLAGS}
        -o "${GPU_EXAMPLE_RAYGEN_SHADER}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/raygen.slang"
    COMMENT "Compiling gpu API example Slang shaders"
    VERBATIM)

add_custom_target(gpu_example_shaders DEPENDS
    "${GPU_EXAMPLE_COMPUTE_SHADER}"
    "${GPU_EXAMPLE_GRAPHICS_SHADER}"
    "${GPU_EXAMPLE_RAYGEN_SHADER}")

function(add_gpu_example target source)
    add_executable(${target} "${source}")
    add_dependencies(${target} gpu_example_shaders)
    target_compile_features(${target} PRIVATE cxx_std_23)
    target_link_libraries(${target} PRIVATE gpu::gpu)
    target_compile_definitions(${target} PRIVATE ${ARGN})
endfunction()

add_gpu_example(gpu_compute_example compute.cpp
    "GPU_COMPUTE_SHADER=\"${GPU_EXAMPLE_COMPUTE_SHADER}\"")
add_gpu_example(gpu_graphics_example graphics.cpp
    "GPU_GRAPHICS_SHADER=\"${GPU_EXAMPLE_GRAPHICS_SHADER}\"")
add_gpu_example(gpu_raytracing_example raytracing.cpp
    "GPU_RAYGEN_SHADER=\"${GPU_EXAMPLE_RAYGEN_SHADER}\"")
