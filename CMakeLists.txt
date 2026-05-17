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
add_subdirectory(external/VulkanMemoryAllocator)

# tinygltf (header-only)
add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE
        ${PROJECT_SOURCE_DIR}/external/tinygltf
)

# tinyobjloader (header-only)
add_library(tinyobjloader INTERFACE)
target_include_directories(tinyobjloader INTERFACE
        ${PROJECT_SOURCE_DIR}/external/tinyobjloader
)

# stb (header-only)
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE
        ${PROJECT_SOURCE_DIR}/external/stb
)

# ImViewGuizmo (header-only)
add_library(ImViewGuizmo INTERFACE)
target_include_directories(ImViewGuizmo INTERFACE
        ${PROJECT_SOURCE_DIR}/external/ImViewGuizmo
)

# Vulkan (Find Vulkan normally)
find_package(Vulkan REQUIRED)

find_program(GLSLC_EXECUTABLE
        NAMES glslc
        HINTS
        "$ENV{VULKAN_SDK}/bin"
        "$ENV{VULKAN_SDK}/Bin"
        REQUIRED
)

set(SHADER_DIR "${PROJECT_SOURCE_DIR}/src/Shaders")
set(SHADER_OUTPUTS
        "${SHADER_DIR}/RTX/RayGeneration.spv"
        "${SHADER_DIR}/RTX/ClosestHit.spv"
        "${SHADER_DIR}/RTX/Miss.spv"
        "${SHADER_DIR}/Compute/PathTracer.spv"
        "${SHADER_DIR}/Tonemapping/Tonemapper.spv"
        "${SHADER_DIR}/Rml/RmlFrag.spv"
        "${SHADER_DIR}/Rml/RmlVert.spv"
        "${SHADER_DIR}/Rml/RmlGradientFrag.spv"
        "${SHADER_DIR}/Rml/RmlFilterFrag.spv"
        "${SHADER_DIR}/Rml/RmlPassthroughFrag.spv"
        "${SHADER_DIR}/Rml/RmlPassthroughVert.spv"
)

add_custom_command(
        OUTPUT ${SHADER_OUTPUTS}
        COMMAND "${GLSLC_EXECUTABLE}" "${SHADER_DIR}/RTX/RayGeneration.glsl" -o "${SHADER_DIR}/RTX/RayGeneration.spv" --target-env=vulkan1.3
        COMMAND "${GLSLC_EXECUTABLE}" "${SHADER_DIR}/RTX/ClosestHit.glsl" -o "${SHADER_DIR}/RTX/ClosestHit.spv" --target-env=vulkan1.3
        COMMAND "${GLSLC_EXECUTABLE}" "${SHADER_DIR}/RTX/Miss.glsl" -o "${SHADER_DIR}/RTX/Miss.spv" --target-env=vulkan1.3
        COMMAND "${GLSLC_EXECUTABLE}" "${SHADER_DIR}/Compute/PathTracer.comp" -o "${SHADER_DIR}/Compute/PathTracer.spv" -DUSE_COMPUTE=1 --target-env=vulkan1.3
        COMMAND "${GLSLC_EXECUTABLE}" "${SHADER_DIR}/Tonemapping/Tonemapper.comp" -o "${SHADER_DIR}/Tonemapping/Tonemapper.spv" --target-env=vulkan1.3
        COMMAND "${GLSLC_EXECUTABLE}" "${SHADER_DIR}/Rml/RmlFrag.frag" -o "${SHADER_DIR}/Rml/RmlFrag.spv" --target-env=vulkan1.3
        COMMAND "${GLSLC_EXECUTABLE}" "${SHADER_DIR}/Rml/RmlVert.vert" -o "${SHADER_DIR}/Rml/RmlVert.spv" --target-env=vulkan1.3
        COMMAND "${GLSLC_EXECUTABLE}" "${SHADER_DIR}/Rml/RmlGradientFrag.frag" -o "${SHADER_DIR}/Rml/RmlGradientFrag.spv" --target-env=vulkan1.3
        COMMAND "${GLSLC_EXECUTABLE}" "${SHADER_DIR}/Rml/RmlFilterFrag.frag" -o "${SHADER_DIR}/Rml/RmlFilterFrag.spv" --target-env=vulkan1.3
        COMMAND "${GLSLC_EXECUTABLE}" "${SHADER_DIR}/Rml/RmlPassthroughFrag.frag" -o "${SHADER_DIR}/Rml/RmlPassthroughFrag.spv" --target-env=vulkan1.3
        COMMAND "${GLSLC_EXECUTABLE}" "${SHADER_DIR}/Rml/RmlPassthroughVert.vert" -o "${SHADER_DIR}/Rml/RmlPassthroughVert.spv" --target-env=vulkan1.3
        DEPENDS
        "${SHADER_DIR}/RTX/RayGeneration.glsl"
        "${SHADER_DIR}/RTX/ClosestHit.glsl"
        "${SHADER_DIR}/RTX/Miss.glsl"
        "${SHADER_DIR}/Compute/PathTracer.comp"
        "${SHADER_DIR}/Tonemapping/Tonemapper.comp"
        "${SHADER_DIR}/Bindings.glsl"
        "${SHADER_DIR}/Common.glsl"
        "${SHADER_DIR}/SharedStructs.h"
        "${SHADER_DIR}/PathTracing/Intersection.glsl"
        "${SHADER_DIR}/PathTracing/PrimaryRayGen.glsl"
        "${SHADER_DIR}/PathTracing/ShadeClosestHit.glsl"
        "${SHADER_DIR}/PathTracing/ShadeMiss.glsl"
        "${SHADER_DIR}/Rml/RmlFrag.frag"
        "${SHADER_DIR}/Rml/RmlVert.vert"
        "${SHADER_DIR}/Rml/RmlGradientFrag.frag"
        "${SHADER_DIR}/Rml/RmlFilterFrag.frag"
        "${SHADER_DIR}/Rml/RmlPassthroughFrag.frag"
        "${SHADER_DIR}/Rml/RmlPassthroughVert.vert"
        COMMENT "Compiling Vulkan shaders"
        VERBATIM
)

add_custom_target(NoorRayShaders DEPENDS ${SHADER_OUTPUTS})

# Lua
file(GLOB LUA_SOURCES
        "${CMAKE_SOURCE_DIR}/external/lua/l*.c"
)
list(FILTER LUA_SOURCES EXCLUDE REGEX ".*lua\\.c$")
list(FILTER LUA_SOURCES EXCLUDE REGEX ".*luac\\.c$")
add_library(lua_from_source STATIC ${LUA_SOURCES})
set_source_files_properties(${LUA_SOURCES} PROPERTIES LANGUAGE CXX)
target_include_directories(lua_from_source PUBLIC
        "${CMAKE_SOURCE_DIR}/external/lua"
)
add_library(Lua::Lua INTERFACE IMPORTED)
target_link_libraries(Lua::Lua INTERFACE lua_from_source)


#  FreeType
add_subdirectory(external/freetype)
add_library(Freetype::Freetype ALIAS freetype)

# LunaSVG
set(LUNASVG_BUILD_EXAMPLES OFF CACHE BOOL "Disable LunaSVG examples" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build LunaSVG as static library" FORCE)
add_subdirectory(external/lunasvg)

# RmlUi
set(RMLUI_SVG_PLUGIN ON CACHE BOOL "Enable SVG plugin" FORCE)
set(RMLUI_FONT_ENGINE "freetype" CACHE STRING "Use Freetype" FORCE)
set(RMLUI_LUA_BINDINGS ON CACHE BOOL "Enable Lua bindings" FORCE)
set(RMLUI_LUA_BINDINGS_LIBRARY "lua_as_cxx" CACHE STRING "Use Lua built as C++" FORCE)
set(RMLUI_SAMPLES OFF CACHE BOOL "Do not build samples" FORCE)
set(RMLUI_TESTS OFF CACHE BOOL "Do not build tests" FORCE)
add_subdirectory(external/rmlui)


# MoltenVK dynamic library (macOS)
if(APPLE AND DEFINED ENV{VULKAN_SDK})
    set(MOLTENVK_DYLIB "$ENV{VULKAN_SDK}/lib/libMoltenVK.dylib")
    if(EXISTS "${MOLTENVK_DYLIB}")
        message(STATUS "Using dynamic MoltenVK library at ${MOLTENVK_DYLIB}")
        add_library(MoltenVK SHARED IMPORTED)
        set_target_properties(MoltenVK PROPERTIES
                IMPORTED_LOCATION "${MOLTENVK_DYLIB}"
                INTERFACE_INCLUDE_DIRECTORIES "$ENV{VULKAN_SDK}/include"
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
        external/imgui/backends/imgui_impl_sdl3.cpp
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
add_dependencies(NoorRay NoorRayShaders)

target_include_directories(NoorRay PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

add_custom_target(NoorRay_RunOnly
        COMMAND NoorRay
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Running NoorRay without rebuilding"
)

# Link against all libraries
target_link_libraries(NoorRay PRIVATE
        portable_file_dialogs
        glm
        imgui
        Vulkan::Vulkan
        SDL3::SDL3-static
        tinygltf
        tinyobjloader
        stb
        ImViewGuizmo
        GPUOpen::VulkanMemoryAllocator

        RmlUi::Core
        RmlUi::Lua
        
        RmlUi::Debugger
)

target_compile_definitions(NoorRay PRIVATE SDL_MAIN_HANDLED)
target_compile_definitions(NoorRay PRIVATE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

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

    # MoltenVk and Vulkan loader bundling and @rpath fix
    if(DEFINED ENV{VULKAN_SDK})
        set(VULKAN_LOADER_DYLIB "$ENV{VULKAN_SDK}/lib/libvulkan.1.dylib")
        set(MOLTENVK_DYLIB "$ENV{VULKAN_SDK}/lib/libMoltenVK.dylib")

        if(EXISTS "${VULKAN_LOADER_DYLIB}" AND EXISTS "${MOLTENVK_DYLIB}")
            add_custom_command(TARGET NoorRay POST_BUILD
                    # Ensure Frameworks folder exists
                    COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/Frameworks"

                    # Copy the Vulkan loader
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${VULKAN_LOADER_DYLIB}"
                    "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/Frameworks/libvulkan.1.dylib"

                    # Copy MoltenVK dylib
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${MOLTENVK_DYLIB}"
                    "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/Frameworks/libMoltenVK.dylib"

                    # Fix binary to use @rpath/libvulkan.1.dylib
                    COMMAND /usr/bin/install_name_tool -change
                    "${VULKAN_LOADER_DYLIB}"
                    "@rpath/libvulkan.1.dylib"
                    "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/MacOS/NoorRay"

                    # Fix binary to use @rpath/libMoltenVK.dylib
                    COMMAND /usr/bin/install_name_tool -change
                    "${MOLTENVK_DYLIB}"
                    "@rpath/libMoltenVK.dylib"
                    "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/MacOS/NoorRay"

                    # Add rpath to point to Frameworks folder
                    COMMAND /usr/bin/install_name_tool -add_rpath
                    "@executable_path/../Frameworks"
                    "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/MacOS/NoorRay"

                    # Fix the dylib IDs
                    COMMAND /usr/bin/install_name_tool -id
                    "@rpath/libvulkan.1.dylib"
                    "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/Frameworks/libvulkan.1.dylib"

                    COMMAND /usr/bin/install_name_tool -id
                    "@rpath/libMoltenVK.dylib"
                    "$<TARGET_BUNDLE_DIR:NoorRay>/Contents/Frameworks/libMoltenVK.dylib"

                    COMMENT "Bundling Vulkan loader and MoltenVK dylib, fixing @rpath for NoorRay"
            )
        else()
            message(FATAL_ERROR "Vulkan loader or MoltenVK dylib not found at expected paths: ${VULKAN_LOADER_DYLIB}, ${MOLTENVK_DYLIB}")
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

if (MINGW)
    message(STATUS "Enabling Static Linking for Release build")
    target_link_options(NoorRay PRIVATE
            -static-libgcc
            -static-libstdc++
            -static
    )
endif()
