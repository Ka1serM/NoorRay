cmake_minimum_required(VERSION 3.10)
project(VulkanToyPathtracer VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMAKE")

# -----------------------------------------------------------------------------
# GLFW, GLM
# -----------------------------------------------------------------------------
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)
add_subdirectory(external/glm)

# -----------------------------------------------------------------------------
# Dear ImGui (core + GLFW+Vulkan backend)
# -----------------------------------------------------------------------------
add_library(imgui STATIC
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    # no demo.cpp if you don’t need it
    # Vulkan + GLFW backends:
    external/imgui/backends/imgui_impl_vulkan.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
)
target_include_directories(imgui PUBLIC
    ${PROJECT_SOURCE_DIR}/external/imgui
    ${PROJECT_SOURCE_DIR}/external/imgui/backends
)
# using GLFW+Vulkan:
target_compile_definitions(imgui PUBLIC 
    IMGUI_IMPL_VULKAN_BINDING
    IMGUI_IMPL_OPENGL_LOADER_GLFW=0
)

# Vulkan headers/libs:
find_package(Vulkan REQUIRED)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

# -----------------------------------------------------------------------------
# main executable
# -----------------------------------------------------------------------------
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.h)
file(GLOB_RECURSE SHADER_FILES src/shaders/*)
file(GLOB_RECURSE ASSET_FILES assets/*)

add_executable(GpuPathtracer ${SRC_FILES})
target_link_libraries(GpuPathtracer PUBLIC
    glfw
    glm
    imgui
    Vulkan::Vulkan
)
target_include_directories(GpuPathtracer PUBLIC
    "$ENV{VULKAN_SDK}/Include"
    "${PROJECT_SOURCE_DIR}/external/tinyobjloader"
    "${PROJECT_SOURCE_DIR}/external/tinygltf"
    "${PROJECT_SOURCE_DIR}/external/stb"
    "${PROJECT_SOURCE_DIR}/src"
)
source_group("src/shaders" FILES ${SHADER_FILES})
source_group("Assets"      FILES ${ASSET_FILES})
set_source_files_properties(${SHADER_FILES} ${ASSET_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
target_sources(GpuPathtracer PRIVATE ${SHADER_FILES} ${ASSET_FILES})
