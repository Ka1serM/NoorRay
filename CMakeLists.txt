cmake_minimum_required(VERSION 3.10)
project(GpuPathtracer VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMAKE")

# GLFW Setup
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

# === Collect all source files in src directory ===
file(GLOB_RECURSE SRC_FILES src/*.cpp)

# === Collect shader files (but don't compile them) ===
file(GLOB_RECURSE SHADER_FILES shaders/*)

# === Collect asset files (but don't compile them) ===
file(GLOB_RECURSE ASSET_FILES assets/*)

# === Create the main executable (GpuPathtracer) ===
add_executable(GpuPathtracer ${SRC_FILES})

# Link libraries and set include directories
target_link_libraries(GpuPathtracer PUBLIC glfw)
target_include_directories(GpuPathtracer PUBLIC
    "$ENV{VULKAN_SDK}/Include"
    "${PROJECT_SOURCE_DIR}/external/tinyobjloader"
    "${PROJECT_SOURCE_DIR}/src"
)

# === Shader compilation target (dynamic) ===
add_custom_target(CompileShaders ALL
    COMMENT "Compiling shaders..."
)

# Loop through each shader file and compile it to SPIR-V
foreach(SHADER_FILE ${SHADER_FILES})
    # Skip files that are already .spv or other shader-related extensions
    if(SHADER_FILE MATCHES ".*\\.(spv|glsl)$")
        continue()
    endif()
    
    # Get the shader name (basename without extension)
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER_FILE} EXT)
    set(SHADER_OUTPUT "${CMAKE_SOURCE_DIR}/shaders/${SHADER_NAME}${SHADER_EXT}.spv")
    
    # Add a custom command for shader compilation using glslc
    add_custom_command(
        TARGET CompileShaders
        COMMAND $ENV{VULKAN_SDK}/Bin/glslc.exe ${SHADER_FILE} -o ${SHADER_OUTPUT} --target-env=vulkan1.3
        DEPENDS ${SHADER_FILE}
        COMMENT "Compiling shader: ${SHADER_NAME}"
    )
endforeach()

# Ensure the main executable depends on shader compilation
add_dependencies(GpuPathtracer CompileShaders)

# This creates a source group for organizing files in Visual Studio's Solution Explorer
source_group("Shaders" FILES ${SHADER_FILES})
source_group("Assets" FILES ${ASSET_FILES})

# For Visual Studio, add the files as resources but not for compilation
set_source_files_properties(${SHADER_FILES} ${ASSET_FILES} PROPERTIES 
    HEADER_FILE_ONLY TRUE  # Mark as header-only so they won't be compiled
)

# Add shader and asset files to IDE only, not for compilation
target_sources(GpuPathtracer PRIVATE ${SHADER_FILES} ${ASSET_FILES})
