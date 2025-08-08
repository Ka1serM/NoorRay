cmake_minimum_required(VERSION 3.16)
project(VulkanToyPathtracer VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 23)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMAKE")

# ---------------------------------------------------------------------------
# SDL3 (Static Linking Setup)
# ---------------------------------------------------------------------------
set(SDL_STATIC ON CACHE BOOL "Build SDL3 as a static library" FORCE)
set(SDL_SHARED OFF CACHE BOOL "Do not build SDL3 as a shared library" FORCE)
add_subdirectory(external/SDL)

# ---------------------------------------------------------------------------
# External Libraries
# ---------------------------------------------------------------------------
add_subdirectory(external/glm)
add_subdirectory(external/portable-file-dialogs)
add_subdirectory(external/json)

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
target_link_libraries(imgui PUBLIC
        Vulkan::Vulkan
        SDL3::SDL3-static
)

# ---------------------------------------------------------------------------
# Main Executable
# ---------------------------------------------------------------------------
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.h)

if(WIN32)
    list(APPEND SRC_FILES resource.rc)
endif()

# The WIN32 keyword is added here to create a GUI-only application
add_executable(VulkanToyPathtracer WIN32 ${SRC_FILES})
set_target_properties(VulkanToyPathtracer PROPERTIES PREFIX "")

target_include_directories(VulkanToyPathtracer PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/external/tinyobjloader
        ${PROJECT_SOURCE_DIR}/external/stb
)
target_link_libraries(VulkanToyPathtracer PRIVATE
        portable_file_dialogs
        glm
        imgui
        Vulkan::Vulkan
        nlohmann_json::nlohmann_json
        SDL3::SDL3-static
)
target_compile_definitions(VulkanToyPathtracer PRIVATE SDL_MAIN_HANDLED)

if(MINGW)
    target_link_options(VulkanToyPathtracer PRIVATE
            -static-libgcc
            -static-libstdc++
            -static
    )
endif()

if(APPLE)
    set_target_properties(VulkanToyPathtracer PROPERTIES MACOSX_BUNDLE TRUE)
    mac_bundle_vulkan_loader(VulkanToyPathtracer)
endif()