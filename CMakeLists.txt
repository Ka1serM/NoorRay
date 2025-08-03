cmake_minimum_required(VERSION 3.16)
project(VulkanToyPathtracer VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 23)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMAKE")

# ---------------------------------------------------------------------------
# External Libraries
# ---------------------------------------------------------------------------
add_subdirectory(external/glm)
add_subdirectory(external/portable-file-dialogs)
add_subdirectory(external/json)

# SDL3
set(SDL_STATIC ON CACHE BOOL "Build SDL3 as a static library")
add_subdirectory(external/SDL)

# Vulkan
find_package(Vulkan REQUIRED)

# ---------------------------------------------------------------------------
# Dear ImGui (core + Vulkan + SDL3 backend) + ImGuizmo
# ---------------------------------------------------------------------------
add_library(imgui STATIC
        external/imgui/imgui.cpp
        external/imgui/imgui_draw.cpp
        external/imgui/imgui_tables.cpp
        external/imgui/imgui_widgets.cpp
        external/imgui/backends/imgui_impl_vulkan.cpp
        external/imgui/backends/imgui_impl_SDL3.cpp
        external/ImGuizmo/ImGuizmo.cpp
)

target_include_directories(imgui PUBLIC
        ${PROJECT_SOURCE_DIR}/external/imgui
        ${PROJECT_SOURCE_DIR}/external/ImGuizmo
)

# FIX: Link to the official SDL3::SDL3 target. This provides both the
# library for the linker and the include paths for the compiler.
target_link_libraries(imgui PUBLIC
        Vulkan::Vulkan
        SDL3::SDL3
)

# ---------------------------------------------------------------------------
# Main Executable
# ---------------------------------------------------------------------------
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.h)
file(GLOB_RECURSE SHADER_FILES src/Shaders/*)
file(GLOB_RECURSE ASSET_FILES assets/*)

add_executable(VulkanToyPathtracer WIN32 ${SRC_FILES})
set_target_properties(VulkanToyPathtracer PROPERTIES PREFIX "")

target_include_directories(VulkanToyPathtracer PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/external/tinyobjloader
        ${PROJECT_SOURCE_DIR}/external/stb
)

# FIX: Link to the official SDL3::SDL3 target here as well.
target_link_libraries(VulkanToyPathtracer PRIVATE
        portable_file_dialogs
        glm
        imgui
        Vulkan::Vulkan
        nlohmann_json::nlohmann_json
        SDL3::SDL3
)

source_group("src/Shaders" FILES ${SHADER_FILES})
source_group("Assets"      FILES ${ASSET_FILES})
set_source_files_properties(${SHADER_FILES} ${ASSET_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
target_sources(VulkanToyPathtracer PRIVATE ${SHADER_FILES} ${ASSET_FILES})

# ---------------------------------------------------------------------------
# Static Linking for Release Builds
# ---------------------------------------------------------------------------
if(MINGW AND CMAKE_BUILD_TYPE STREQUAL "Release")
    target_link_options(VulkanToyPathtracer PRIVATE
            -static-libgcc
            -static-libstdc++
            -Wl,-Bstatic
            -Wl,--whole-archive -lwinpthread -Wl,--no-whole-archive
            -Wl,-Bdynamic
    )
endif()