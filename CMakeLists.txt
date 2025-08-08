cmake_minimum_required(VERSION 3.16)
project(VulkanToyPathtracer VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMAKE")

if(APPLE)
    # Set minimum macOS version to 15.5 (includes MTLResidencySetDescriptor)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "15.5" CACHE STRING "Minimum macOS deployment version")

    # Use SDKROOT environment variable if set, fallback to system default
    if(DEFINED ENV{SDKROOT})
        set(CMAKE_OSX_SYSROOT $ENV{SDKROOT} CACHE PATH "SDK root path")
    endif()
endif()

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

# Vulkan (Find Vulkan normally)
find_package(Vulkan REQUIRED)

# ---------------------------------------------------------------------------
# MoltenVK static library from Vulkan SDK .xcframework on macOS
# ---------------------------------------------------------------------------
if(APPLE AND DEFINED ENV{VULKAN_SDK})
    set(MOLTENVK_LIB "$ENV{VULKAN_SDK}/lib/MoltenVK.xcframework/macos-arm64_x86_64/libMoltenVK.a")
    if(EXISTS "${MOLTENVK_LIB}")
        message(STATUS "Using static MoltenVK library at ${MOLTENVK_LIB}")
        add_library(MoltenVK STATIC IMPORTED)
        set_target_properties(MoltenVK PROPERTIES
                IMPORTED_LOCATION "${MOLTENVK_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "$ENV{VULKAN_SDK}/include"
        )
    else()
        message(FATAL_ERROR "MoltenVK static library not found at expected path: ${MOLTENVK_LIB}")
    endif()
endif()

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
# Gather source files
# ---------------------------------------------------------------------------
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.h)

# ---------------------------------------------------------------------------
# Resource handling on Windows (kept as is)
# ---------------------------------------------------------------------------
if(WIN32)
    set(RESOURCE_RC_PATH "${CMAKE_BINARY_DIR}/resource.rc")

    file(WRITE "${RESOURCE_RC_PATH}"
            "MAINICON ICON \"${PROJECT_SOURCE_DIR}/assets/icon.ico\"\n"
    )

    message(STATUS "Generated resource.rc at ${RESOURCE_RC_PATH}")

    list(APPEND SRC_FILES "${RESOURCE_RC_PATH}")
endif()

# ---------------------------------------------------------------------------
# Executable target
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Platform specific linking and settings
# ---------------------------------------------------------------------------
if(APPLE)
    target_link_libraries(VulkanToyPathtracer PRIVATE MoltenVK)

    target_link_libraries(VulkanToyPathtracer PRIVATE
            "-framework Metal"
            "-framework CoreVideo"
            "-framework Cocoa"
            "-framework QuartzCore"
            "-framework CoreGraphics"
            "-framework IOSurface"
    )

    set_target_properties(VulkanToyPathtracer PROPERTIES MACOSX_BUNDLE TRUE)

    # Minimal Info.plist inside app bundle
    set(APP_BUNDLE_DIR "${CMAKE_BINARY_DIR}/VulkanToyPathtracer.app")
    set(INFO_PLIST_CONTENT "<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
    <key>CFBundleName</key>
    <string>VulkanToyPathtracer</string>
    <key>CFBundleExecutable</key>
    <string>VulkanToyPathtracer</string>
    <key>CFBundleIdentifier</key>
    <string>dev.marcelk.VulkanToyPathtracer</string>
    <key>CFBundleVersion</key>
    <string>${PROJECT_VERSION}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
</dict>
</plist>")
    file(WRITE "${APP_BUNDLE_DIR}/Contents/Info.plist" "${INFO_PLIST_CONTENT}")
endif()

if(MINGW)
    target_link_options(VulkanToyPathtracer PRIVATE
            -static-libgcc
            -static-libstdc++
            -static
    )
endif()
