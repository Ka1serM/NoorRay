cmake_minimum_required(VERSION 3.16)
project(NoorRay VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMAKE")

# SDL3 (Static Linking Setup)
set(SDL_STATIC ON CACHE BOOL "Build SDL3 as a static library" FORCE)
set(SDL_SHARED OFF CACHE BOOL "Do not build SDL3 as a shared library" FORCE)
add_subdirectory(external/SDL)

# External Libraries
add_subdirectory(external/glm)
add_subdirectory(external/portable-file-dialogs)

# tinygltf (header-only)
add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE
        ${PROJECT_SOURCE_DIR}/external/tinygltf
)

# Vulkan (Find Vulkan normally)
find_package(Vulkan REQUIRED)

# MoltenVK dynamic library (macOS)
if(APPLE AND DEFINED ENV{VULKAN_SDK})
    set(MOLTENVK_DYLIB "$ENV{VULKAN_SDK}/macOS/lib/libMoltenVK.dylib")
    if(EXISTS "${MOLTENVK_DYLIB}")
        message(STATUS "Using dynamic MoltenVK library at ${MOLTENVK_DYLIB}")
        add_library(MoltenVK SHARED IMPORTED)
        set_target_properties(MoltenVK PROPERTIES
                IMPORTED_LOCATION "${MOLTENVK_DYLIB}"
                INTERFACE_INCLUDE_DIRECTORIES "$ENV{VULKAN_SDK}/macOS/include"
        )
    else()
        message(FATAL_ERROR "MoltenVK dylib not found at expected path: ${MOLTENVK_DYLIB}")
    endif()
endif()

# Dear ImGui (core + Vulkan + SDL3 backend) + ImGuizmo
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

# Gather source files
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.h)

# Resource handling on Windows
if(WIN32)
    set(RESOURCE_RC_PATH "${CMAKE_BINARY_DIR}/resource.rc")
    file(WRITE "${RESOURCE_RC_PATH}"
            "MAINICON ICON \"${PROJECT_SOURCE_DIR}/assets/icons/icon.ico\"\n"
    )
    message(STATUS "Generated resource.rc at ${RESOURCE_RC_PATH}")
    list(APPEND SRC_FILES "${RESOURCE_RC_PATH}")
endif()

# Executable target
add_executable(NoorRay WIN32 ${SRC_FILES})
set_target_properties(NoorRay PROPERTIES PREFIX "")

target_include_directories(NoorRay PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/external/tinyobjloader
        ${PROJECT_SOURCE_DIR}/external/stb
)

target_link_libraries(NoorRay PRIVATE
        portable_file_dialogs
        glm
        imgui
        Vulkan::Vulkan
        SDL3::SDL3-static
        tinygltf
)

target_compile_definitions(NoorRay PRIVATE SDL_MAIN_HANDLED)

# Platform specific linking and settings
if(APPLE)
    target_link_libraries(NoorRay PRIVATE MoltenVK)

    target_link_libraries(NoorRay PRIVATE
            "-framework Metal"
            "-framework CoreVideo"
            "-framework Cocoa"
            "-framework QuartzCore"
            "-framework CoreGraphics"
            "-framework IOSurface"
    )

    set_target_properties(NoorRay PROPERTIES
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_IDENTIFIER "dev.marcelk.noorray"
            MACOSX_BUNDLE_NAME "NoorRay"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}"
            MACOSX_BUNDLE_VERSION "${PROJECT_VERSION}"
            MACOSX_BUNDLE_COPYRIGHT "Copyright © 2025 Marcel K. All rights reserved."
            MACOSX_BUNDLE_ICON_FILE "icon.icns"
    )

    # Vulkan loader bundling and @rpath fix
    if(DEFINED ENV{VULKAN_SDK})
        set(VULKAN_LOADER_DYLIB "$ENV{VULKAN_SDK}/macOS/lib/libvulkan.1.dylib")
        if(EXISTS "${VULKAN_LOADER_DYLIB}")
            add_custom_command(TARGET NoorRay POST_BUILD
                    # Ensure Frameworks folder exists
                    COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/Frameworks"

                    # Copy the Vulkan loader
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${VULKAN_LOADER_DYLIB}"
                    "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/Frameworks/libvulkan.1.dylib"

                    # Fix binary to use @rpath/libvulkan.1.dylib
                    COMMAND /usr/bin/install_name_tool -change
                    "${VULKAN_LOADER_DYLIB}"
                    "@rpath/libvulkan.1.dylib"
                    "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/MacOS/NoorRay"

                    # Add rpath to point to Frameworks folder
                    COMMAND /usr/bin/install_name_tool -add_rpath
                    "@executable_path/../Frameworks"
                    "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/MacOS/NoorRay"

                    # Fix the dylib's own ID
                    COMMAND /usr/bin/install_name_tool -id
                    "@rpath/libvulkan.1.dylib"
                    "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/Frameworks/libvulkan.1.dylib"

                    COMMENT "Bundling Vulkan loader and fixing @rpath for NoorRay"
            )
        else()
            message(FATAL_ERROR "Vulkan loader not found at ${VULKAN_LOADER_DYLIB}")
        endif()
    endif()

    # Resources handling
    add_custom_command(TARGET NoorRay POST_BUILD
        # Create Resource folder chain
        COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/Resources/vulkan/icd.d"
    
        # Copy app icon
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/assets/icons/icon.icns"
        "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/Resources/icon.icns"
        
        # Copy MoltenVK_icd.json
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/assets/MoltenVK_icd.json"
        "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/Resources/vulkan/icd.d/MoltenVK_icd.json"
    
        COMMENT "Copying app icon and MoltenVK ICD JSON into Resources"
    )


endif()

# Use sanitizers ONLY for Debug builds
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Enabling Sanitizers for Debug build")
    add_compile_options(-fsanitize=address -fsanitize=undefined)
    add_link_options(-fsanitize=address -fsanitize=undefined)
endif()

if (MINGW)
    message(STATUS "Enabling Static Linking for Release build")
    target_link_options(NoorRay PRIVATE
            -static-libgcc
            -static-libstdc++
            -static
    )
endif()
