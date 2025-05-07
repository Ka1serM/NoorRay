cmake_minimum_required(VERSION 3.10)
project(GpuPathtracer VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMAKE")

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

# GLM
add_subdirectory(external/glm)

# === Collect all source files in src directory ===
file(GLOB_RECURSE SRC_FILES
        src/*.cpp
        src/*.h
)

# === Collect shader files from the src/shaders directory (do not compile) ===
file(GLOB_RECURSE SHADER_FILES src/shaders/*)

# === Collect asset files (do not compile) ===
file(GLOB_RECURSE ASSET_FILES assets/*)

# === Create the main executable (GpuPathtracer) ===
add_executable(GpuPathtracer ${SRC_FILES})

# Link libraries
target_link_libraries(GpuPathtracer
        PUBLIC
        glfw
        glm
)

# Include Directories
target_include_directories(GpuPathtracer PUBLIC
        "$ENV{VULKAN_SDK}/Include"
        "${PROJECT_SOURCE_DIR}/external/tinyobjloader"
        "${PROJECT_SOURCE_DIR}/external/tinygltf"
        "${PROJECT_SOURCE_DIR}/external/stb"
        "${PROJECT_SOURCE_DIR}/src"
)

# This creates a source group for organizing files in Visual Studio's Solution Explorer
source_group("src/shaders" FILES ${SHADER_FILES})
source_group("Assets" FILES ${ASSET_FILES})

# For Visual Studio, add the files as resources but not for compilation
set_source_files_properties(${SHADER_FILES} ${ASSET_FILES} PROPERTIES
        HEADER_FILE_ONLY TRUE  # Mark as header-only so they won't be compiled
)

# Add shader and asset files to IDE only, not for compilation
target_sources(GpuPathtracer PRIVATE ${SHADER_FILES} ${ASSET_FILES})