cmake_minimum_required(VERSION 3.25)
project(NoorRay VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMAKE")

find_package(CUDAToolkit 13.3 EXACT REQUIRED)
set(OPTIX_ROOT "/home/marcel/Programs/OptixSDK" CACHE PATH "OptiX 9.1 installation")
find_path(OPTIX_INCLUDE_DIR optix.h HINTS "${OPTIX_ROOT}/include" REQUIRED)
file(STRINGS "${OPTIX_INCLUDE_DIR}/optix.h" OPTIX_VERSION_LINE
        REGEX "^#define OPTIX_VERSION 90100$")
if(NOT OPTIX_VERSION_LINE)
    message(FATAL_ERROR "NoorRay requires OptiX 9.1 headers")
endif()

# Suppress verbose status output from all external dependencies
set(CMAKE_MESSAGE_LOG_LEVEL WARNING)
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

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

# Restore normal log level for project-own messages
set(CMAKE_MESSAGE_LOG_LEVEL STATUS)

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

# Base slangc flags (software BVH, no USE_RAY_QUERY)
set(SLANGC_BASE
        -target spirv
        -profile spirv_1_5
        -emit-spirv-directly
        -matrix-layout-column-major
        -fvk-use-entrypoint-name
        -Wno-30081
)

set(SHADER_OUTPUTS
        "${SHADER_DIR}/Tonemapping/Tonemapper.spv"
)

add_custom_command(
        OUTPUT ${SHADER_OUTPUTS}
        COMMAND "${SLANGC_EXECUTABLE}" "${SHADER_DIR}/Tonemapping/Tonemapper.slang" -entry main ${SLANGC_BASE} -o "${SHADER_DIR}/Tonemapping/Tonemapper.spv"
        DEPENDS
        "${SHADER_DIR}/Tonemapping/Tonemapper.slang"
        COMMENT "Compiling the Vulkan tonemapper shader"
        VERBATIM
)

add_custom_target(NoorRayShaders DEPENDS ${SHADER_OUTPUTS})

# Generate embed wrapper headers with absolute paths (GCC 15 #embed only resolves
# absolute or source-relative paths, not -I search paths)
cmake_path(SET NR_GEN "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${NR_GEN}")

file(CONFIGURE OUTPUT "${NR_GEN}/TonemapperSpv.h" CONTENT
"#pragma once
static constexpr unsigned char noorRayTonemapperSpv[] = {
    #embed \"${SHADER_DIR}/Tonemapping/Tonemapper.spv\"
};
static constexpr size_t noorRayTonemapperSpvLength = sizeof(noorRayTonemapperSpv);\n" @ONLY)

file(CONFIGURE OUTPUT "${NR_GEN}/ImGuiFont.h" CONTENT
"#pragma once
static constexpr unsigned char noorRayImGuiFont[] = {
    #embed \"${PROJECT_SOURCE_DIR}/assets/fonts/Inter.ttf\"
};
static constexpr size_t noorRayImGuiFontLength = sizeof(noorRayImGuiFont);\n" @ONLY)

file(CONFIGURE OUTPUT "${NR_GEN}/ImGuiLayout.h" CONTENT
"#pragma once
static constexpr unsigned char noorRayImGuiLayout[] = {
    #embed \"${PROJECT_SOURCE_DIR}/assets/imgui.ini\"
};
static constexpr size_t noorRayImGuiLayoutLength = sizeof(noorRayImGuiLayout);\n" @ONLY)

# NR_OPTIX_PTX defined here early so it can be used in the embed header and
# also referenced by the custom command further below
set(NR_OPTIX_PTX "${CMAKE_CURRENT_BINARY_DIR}/generated/NoorRayOptix.ptx")

file(CONFIGURE OUTPUT "${NR_GEN}/NoorRayOptixIr.h" CONTENT
"#pragma once
static constexpr unsigned char noorRayOptixIr[] = {
    #embed \"${NR_OPTIX_PTX}\"
};
static constexpr size_t noorRayOptixIrLength = sizeof(noorRayOptixIr);\n" @ONLY)

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
list(FILTER SRC_FILES EXCLUDE REGEX "/GPU/")
list(FILTER SRC_FILES EXCLUDE REGEX "/Kernels/cuda/")
list(FILTER SRC_FILES EXCLUDE REGEX "/Raytracing/GpuWavefrontRaytracer\\.cpp$")

file(GLOB_RECURSE NR_KERNEL_HEADERS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/Kernels/*.h"
        "${PROJECT_SOURCE_DIR}/src/Kernels/*/*.h"
        "${PROJECT_SOURCE_DIR}/src/GPU/Annotations.h"
        "${PROJECT_SOURCE_DIR}/src/GPU/RayTraversal.h")

add_custom_command(
        OUTPUT "${NR_OPTIX_PTX}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_CUDA_COMPILER}"
                -ccbin "${CMAKE_CUDA_HOST_COMPILER}"
                --ptx --gpu-architecture=compute_89 --std=c++20
                --use_fast_math --generate-line-info --expt-relaxed-constexpr
                -DNR_BACKEND_CUDA=1 -DGLM_FORCE_CUDA=1
                "-I${PROJECT_SOURCE_DIR}/src"
                "-I${PROJECT_SOURCE_DIR}/external/glm"
                "-I${OPTIX_INCLUDE_DIR}"
                -o "${NR_OPTIX_PTX}"
                "${PROJECT_SOURCE_DIR}/src/Kernels/cuda/OptixPrograms.cu"
        DEPENDS
                src/Kernels/cuda/OptixPrograms.cu
                ${NR_KERNEL_HEADERS}
        VERBATIM)
add_custom_target(NoorRayOptixIr DEPENDS "${NR_OPTIX_PTX}")

# .h files compiled with NVCC as CUDA translation units
set(NR_CUDA_KERNEL_SOURCES
        src/Kernels/Generate.h
        src/Kernels/Shade.h
        src/Kernels/Finalize.h)
set_source_files_properties(${NR_CUDA_KERNEL_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(NoorRayCuda STATIC
        src/GPU/CudaDevice.cpp
        src/GPU/ImageInterop.cpp
        src/GPU/Memory.cpp
        src/Kernels/cuda/CudaRayLutGenerator.cu
        ${NR_CUDA_KERNEL_SOURCES}
        src/Kernels/cuda/AccelBuilder.cpp
        src/Kernels/cuda/OptiXSetup.cpp
        src/Raytracing/GpuWavefrontRaytracer.cpp)
add_dependencies(NoorRayCuda NoorRayOptixIr)
set_target_properties(NoorRayCuda PROPERTIES CUDA_ARCHITECTURES 89)
target_compile_features(NoorRayCuda PUBLIC cxx_std_20 cuda_std_20)
target_compile_definitions(NoorRayCuda PUBLIC
        NR_BACKEND_CUDA=1
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
target_compile_definitions(NoorRayCuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:GLM_FORCE_CUDA=1>)
target_include_directories(NoorRayCuda PUBLIC
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/external/glm"
        "${OPTIX_INCLUDE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_options(NoorRayCuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--generate-line-info;--extended-lambda;--expt-relaxed-constexpr>)
target_link_libraries(NoorRayCuda PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        glm
        stb
        Vulkan::Vulkan
        SDL3::SDL3-static
        GPUOpen::VulkanMemoryAllocator)

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
        ${CMAKE_CURRENT_BINARY_DIR}/generated
)

# Link against all libraries
target_link_libraries(NoorRay PRIVATE
        NoorRayCuda
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

target_compile_definitions(NoorRay PRIVATE SDL_MAIN_HANDLED NR_BACKEND_CUDA=1)
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
