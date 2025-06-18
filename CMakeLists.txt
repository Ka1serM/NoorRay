cmake_minimum_required(VERSION 3.10)
project(UniversalUmap.Rendering VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 23)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMAKE")

# ---------------------------------------------------------------------------
# GLFW, GLM
# ---------------------------------------------------------------------------
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)
add_subdirectory(external/glm)
add_subdirectory(external/portable-file-dialogs)

# ---------------------------------------------------------------------------
# Dear ImGui (core + GLFW+Vulkan backend)
# ---------------------------------------------------------------------------
add_library(imgui STATIC
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    # Vulkan + GLFW backends:
    external/imgui/backends/imgui_impl_vulkan.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
)
target_include_directories(imgui PUBLIC
    ${PROJECT_SOURCE_DIR}/external/imgui
    ${PROJECT_SOURCE_DIR}/external/imgui/backends
)
target_compile_definitions(imgui PUBLIC 
    IMGUI_IMPL_VULKAN_BINDING
    IMGUI_IMPL_OPENGL_LOADER_GLFW=0
)

# Vulkan headers/libs:
find_package(Vulkan REQUIRED)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

# ---------------------------------------------------------------------------
# main DLL library
# ---------------------------------------------------------------------------
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.h)
file(GLOB_RECURSE SHADER_FILES src/Shaders/*)
file(GLOB_RECURSE ASSET_FILES assets/*)

add_executable(UniversalUmap.Rendering WIN32 ${SRC_FILES})
#add_library(UniversalUmap.Rendering SHARED ${SRC_FILES})
set_target_properties(UniversalUmap.Rendering PROPERTIES PREFIX "")

target_link_libraries(UniversalUmap.Rendering PUBLIC
    portable_file_dialogs
    glfw
    glm
    imgui
    Vulkan::Vulkan
)

target_include_directories(UniversalUmap.Rendering PUBLIC
    "$ENV{VULKAN_SDK}/Include"
    "${PROJECT_SOURCE_DIR}/external/tinyobjloader"
    "${PROJECT_SOURCE_DIR}/external/tinygltf"
    "${PROJECT_SOURCE_DIR}/external/stb"
    "${PROJECT_SOURCE_DIR}/src"
)
source_group("src/Shaders" FILES ${SHADER_FILES})
source_group("Assets"      FILES ${ASSET_FILES})
set_source_files_properties(${SHADER_FILES} ${ASSET_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
target_sources(UniversalUmap.Rendering PRIVATE ${SHADER_FILES} ${ASSET_FILES})

# ----------------------------------------------------
# Add static linking flags for Release builds
# ----------------------------------------------------
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_link_options(UniversalUmap.Rendering PRIVATE
            -static-libgcc
            -static-libstdc++
            -Wl,-Bstatic
            -Wl,--whole-archive -lwinpthread -Wl,--no-whole-archive
            -Wl,-Bdynamic
    )
endif()
