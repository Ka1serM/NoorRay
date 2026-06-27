cmake_minimum_required(VERSION 3.16)
project(NoorRay VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)
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

# glaze (header-only JSON/reflection library)
add_library(glaze INTERFACE)
target_include_directories(glaze INTERFACE
        ${PROJECT_SOURCE_DIR}/external/glaze/include
)

# Intel oneTBB
set(TBB_TEST     OFF CACHE BOOL "" FORCE)
set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TBB_STRICT   OFF CACHE BOOL "" FORCE)
add_subdirectory(external/tbb)

# Vulkan
if(DEFINED ENV{VULKAN_SDK})
    set(VULKAN_SDK_ROOT "$ENV{VULKAN_SDK}")
    if(NOT EXISTS "${VULKAN_SDK_ROOT}/include")
        get_filename_component(VULKAN_SDK_PARENT "${VULKAN_SDK_ROOT}" DIRECTORY)
        if(EXISTS "${VULKAN_SDK_PARENT}/include")
            set(VULKAN_SDK_ROOT "${VULKAN_SDK_PARENT}")
        endif()
    endif()

    set(Vulkan_INCLUDE_DIR "${VULKAN_SDK_ROOT}/include")
    if(EXISTS "${VULKAN_SDK_ROOT}/lib/VulkanLoader/lib/libvulkan.so")
        set(Vulkan_LIBRARY "${VULKAN_SDK_ROOT}/lib/VulkanLoader/lib/libvulkan.so")
    elseif(EXISTS "${VULKAN_SDK_ROOT}/lib/libvulkan.so")
        set(Vulkan_LIBRARY "${VULKAN_SDK_ROOT}/lib/libvulkan.so")
    endif()
endif()
find_package(Vulkan REQUIRED)

find_program(SLANGC_EXECUTABLE
        NAMES slangc
        HINTS
        "${VULKAN_SDK_ROOT}/bin"
        "$ENV{VULKAN_SDK}/bin"
        REQUIRED
)

set(SHADER_DIR     "${PROJECT_SOURCE_DIR}/src/Shaders")
set(WAVEFRONT_DIR  "${SHADER_DIR}/Wavefront")

# Shared Slang source dependencies (all UU shader files)
set(SLANG_DEPS
        "${SHADER_DIR}/Shared.h"
        "${SHADER_DIR}/Bindings.slang"
        "${SHADER_DIR}/Bxdfs/Brdf.slang"
        "${SHADER_DIR}/Bxdfs/IBxdf.slang"
        "${SHADER_DIR}/Bxdfs/PbrBxdf.slang"
        "${SHADER_DIR}/Cameras/FisheyeCamera.slang"
        "${SHADER_DIR}/Cameras/ICamera.slang"
        "${SHADER_DIR}/Cameras/OrthographicCamera.slang"
        "${SHADER_DIR}/Cameras/PerspectiveCamera.slang"
        "${SHADER_DIR}/Cameras/RealisticCamera.slang"
        "${SHADER_DIR}/Cameras/ThinLensCamera.slang"
        "${SHADER_DIR}/Core/Accumulation.slang"
        "${SHADER_DIR}/Core/AdaptiveSampling.slang"
        "${SHADER_DIR}/Core/Core.slang"
        "${SHADER_DIR}/Core/Payload.slang"
        "${SHADER_DIR}/Core/RenderPolicy.slang"
        "${SHADER_DIR}/Geometry/Intersection.slang"
        "${SHADER_DIR}/Geometry/SurfaceData.slang"
        "${SHADER_DIR}/Lights/DirectionalLight.slang"
        "${SHADER_DIR}/Lights/EnvironmentLight.slang"
        "${SHADER_DIR}/Lights/ILight.slang"
        "${SHADER_DIR}/Lights/SceneLight.slang"
        "${SHADER_DIR}/Materials/PbrMaterial.slang"
        "${SHADER_DIR}/Samplers/HaltonSampler.slang"
        "${SHADER_DIR}/Samplers/ISampler.slang"
        "${SHADER_DIR}/Samplers/R2Sampler.slang"
        "${SHADER_DIR}/Common.slang"
        "${SHADER_DIR}/Integrators.slang"
        "${WAVEFRONT_DIR}/Generate.slang"
        "${WAVEFRONT_DIR}/Extend.slang"
        "${WAVEFRONT_DIR}/Shade.slang"
        "${WAVEFRONT_DIR}/Connect.slang"
        "${WAVEFRONT_DIR}/Finalize.slang"
        "${WAVEFRONT_DIR}/Advance.slang"
        "${WAVEFRONT_DIR}/GenerateRayLut.slang"
)

# Base slangc flags (software BVH, no USE_RAY_QUERY)
set(SLANGC_BASE
        -target spirv
        -profile spirv_1_5
        -emit-spirv-directly
        -matrix-layout-column-major
        -fvk-use-entrypoint-name
        -Wno-30081
)

# RTX variant adds USE_RAY_QUERY and the ray-query capability
set(SLANGC_RTX ${SLANGC_BASE} -DUSE_RAY_QUERY=1 -capability rayquery)

set(SHADER_OUTPUTS
        "${SHADER_DIR}/Tonemapping/Tonemapper.spv"
        "${WAVEFRONT_DIR}/Generate.spv"
        "${WAVEFRONT_DIR}/Extend.spv"
        "${WAVEFRONT_DIR}/Shade.spv"
        "${WAVEFRONT_DIR}/Connect.spv"
        "${WAVEFRONT_DIR}/Finalize.spv"
        "${WAVEFRONT_DIR}/Advance.spv"
        "${WAVEFRONT_DIR}/GenerateRayLut.spv"
        "${WAVEFRONT_DIR}/GenerateRtx.spv"
        "${WAVEFRONT_DIR}/ExtendRtx.spv"
        "${WAVEFRONT_DIR}/ShadeRtx.spv"
        "${WAVEFRONT_DIR}/ConnectRtx.spv"
        "${WAVEFRONT_DIR}/FinalizeRtx.spv"
        "${WAVEFRONT_DIR}/AdvanceRtx.spv"
)

add_custom_command(
        OUTPUT ${SHADER_OUTPUTS}
        # Tonemapper
        COMMAND "${SLANGC_EXECUTABLE}" "${SHADER_DIR}/Tonemapping/Tonemapper.slang" -entry main ${SLANGC_BASE} -o "${SHADER_DIR}/Tonemapping/Tonemapper.spv"
        # Compute wavefront passes (software BVH, no USE_RAY_QUERY)
        COMMAND "${SLANGC_EXECUTABLE}" "${WAVEFRONT_DIR}/Generate.slang" -entry main ${SLANGC_BASE} -o "${WAVEFRONT_DIR}/Generate.spv"
        COMMAND "${SLANGC_EXECUTABLE}" "${WAVEFRONT_DIR}/Extend.slang"   -entry main ${SLANGC_BASE} -o "${WAVEFRONT_DIR}/Extend.spv"
        COMMAND "${SLANGC_EXECUTABLE}" "${WAVEFRONT_DIR}/Shade.slang"    -entry main ${SLANGC_BASE} -o "${WAVEFRONT_DIR}/Shade.spv"
        COMMAND "${SLANGC_EXECUTABLE}" "${WAVEFRONT_DIR}/Connect.slang"  -entry main ${SLANGC_BASE} -o "${WAVEFRONT_DIR}/Connect.spv"
        COMMAND "${SLANGC_EXECUTABLE}" "${WAVEFRONT_DIR}/Finalize.slang" -entry main ${SLANGC_BASE} -o "${WAVEFRONT_DIR}/Finalize.spv"
        COMMAND "${SLANGC_EXECUTABLE}" "${WAVEFRONT_DIR}/Advance.slang"  -entry main ${SLANGC_BASE} -o "${WAVEFRONT_DIR}/Advance.spv"
        COMMAND "${SLANGC_EXECUTABLE}" "${WAVEFRONT_DIR}/GenerateRayLut.slang" -entry main ${SLANGC_BASE} -o "${WAVEFRONT_DIR}/GenerateRayLut.spv"
        # RTX wavefront passes (inline RayQuery, USE_RAY_QUERY=1)
        COMMAND "${SLANGC_EXECUTABLE}" "${WAVEFRONT_DIR}/Generate.slang" -entry main ${SLANGC_RTX} -o "${WAVEFRONT_DIR}/GenerateRtx.spv"
        COMMAND "${SLANGC_EXECUTABLE}" "${WAVEFRONT_DIR}/Extend.slang"   -entry main ${SLANGC_RTX} -o "${WAVEFRONT_DIR}/ExtendRtx.spv"
        COMMAND "${SLANGC_EXECUTABLE}" "${WAVEFRONT_DIR}/Shade.slang"    -entry main ${SLANGC_RTX} -o "${WAVEFRONT_DIR}/ShadeRtx.spv"
        COMMAND "${SLANGC_EXECUTABLE}" "${WAVEFRONT_DIR}/Connect.slang"  -entry main ${SLANGC_RTX} -o "${WAVEFRONT_DIR}/ConnectRtx.spv"
        COMMAND "${SLANGC_EXECUTABLE}" "${WAVEFRONT_DIR}/Finalize.slang" -entry main ${SLANGC_RTX} -o "${WAVEFRONT_DIR}/FinalizeRtx.spv"
        COMMAND "${SLANGC_EXECUTABLE}" "${WAVEFRONT_DIR}/Advance.slang"  -entry main ${SLANGC_RTX} -o "${WAVEFRONT_DIR}/AdvanceRtx.spv"
        DEPENDS
        "${SHADER_DIR}/Tonemapping/Tonemapper.slang"
        ${SLANG_DEPS}
        COMMENT "Compiling Slang shaders (Tonemapper + wavefront variants + RayLUT generator)"
        VERBATIM
)

add_custom_target(NoorRayShaders DEPENDS ${SHADER_OUTPUTS})



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

# Executable target — WIN32 only applies on Windows (suppresses the console window
# for GUI mode), but CLI mode still works because SDL_MAIN_HANDLED is set.
if(WIN32)
    add_executable(NoorRay WIN32 ${SRC_FILES})
else()
    add_executable(NoorRay ${SRC_FILES})
endif()
set_target_properties(NoorRay PROPERTIES PREFIX "")
add_dependencies(NoorRay NoorRayShaders)

target_include_directories(NoorRay PRIVATE
        ${PROJECT_SOURCE_DIR}/src
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
        glaze
        TBB::tbb
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
