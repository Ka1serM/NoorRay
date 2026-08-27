set(NR_GPU_EXAMPLE_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(NR_GPU_EXAMPLE_COMPUTE_SHADER "${NR_GPU_EXAMPLE_SHADER_DIR}/compute.spv")
set(NR_GPU_EXAMPLE_GRAPHICS_SHADER "${NR_GPU_EXAMPLE_SHADER_DIR}/triangle.spv")
set(NR_GPU_EXAMPLE_RAYGEN_SHADER "${NR_GPU_EXAMPLE_SHADER_DIR}/raygen.spv")

add_custom_command(
    OUTPUT "${NR_GPU_EXAMPLE_COMPUTE_SHADER}"
           "${NR_GPU_EXAMPLE_GRAPHICS_SHADER}"
           "${NR_GPU_EXAMPLE_RAYGEN_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${NR_GPU_EXAMPLE_SHADER_DIR}"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute.slang"
        -entry main -stage compute -target spirv
        -profile spirv_1_5+SPV_EXT_descriptor_heap
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -spirv-resource-heap-stride 128
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_GPU_EXAMPLE_COMPUTE_SHADER}"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.slang"
        -entry vertMain -stage vertex -entry fragMain -stage fragment
        -target spirv -profile spirv_1_5+SPV_EXT_descriptor_heap
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -spirv-resource-heap-stride 128
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_GPU_EXAMPLE_GRAPHICS_SHADER}"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/raygen.slang"
        -entry main -stage raygeneration -target spirv
        -profile spirv_1_5+spvRayTracingKHR+SPV_EXT_descriptor_heap
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -spirv-resource-heap-stride 128
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_GPU_EXAMPLE_RAYGEN_SHADER}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/raygen.slang"
    COMMENT "Compiling gpu API example Slang shaders"
    VERBATIM)

add_custom_target(gpu_api_example_shaders DEPENDS
    "${NR_GPU_EXAMPLE_COMPUTE_SHADER}"
    "${NR_GPU_EXAMPLE_GRAPHICS_SHADER}"
    "${NR_GPU_EXAMPLE_RAYGEN_SHADER}")

function(add_gpu_api_example target source shader_macro shader_path)
    add_executable(${target} "${source}")
    add_dependencies(${target} gpu_api_example_shaders)
    target_compile_features(${target} PRIVATE cxx_std_23)
    target_include_directories(${target} PRIVATE
        "${PROJECT_SOURCE_DIR}/src/gpu/include")
    target_compile_definitions(${target} PRIVATE
        "${shader_macro}=\"${shader_path}\"")
    target_link_libraries(${target} PRIVATE noorray_gpu)
endfunction()

add_gpu_api_example(gpu_compute_example compute.cpp
    GPU_COMPUTE_SHADER "${NR_GPU_EXAMPLE_COMPUTE_SHADER}")

add_executable(gpu_graphics_example graphics.cpp)
add_dependencies(gpu_graphics_example gpu_api_example_shaders)
target_compile_features(gpu_graphics_example PRIVATE cxx_std_23)
target_include_directories(gpu_graphics_example PRIVATE
    "${PROJECT_SOURCE_DIR}/src/gpu/include")
target_compile_definitions(gpu_graphics_example PRIVATE
    "GPU_GRAPHICS_SHADER=\"${NR_GPU_EXAMPLE_GRAPHICS_SHADER}\"")
target_link_libraries(gpu_graphics_example PRIVATE noorray_gpu)

add_gpu_api_example(gpu_raytracing_example raytracing.cpp
    GPU_RAYGEN_SHADER "${NR_GPU_EXAMPLE_RAYGEN_SHADER}")
