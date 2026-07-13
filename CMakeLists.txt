cmake_minimum_required(VERSION 3.25)
project(NoorRay VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_compile_options($<$<CONFIG:Release>:-g>)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build dependencies as static libraries" FORCE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
option(NR_ENABLE_UNITY_BUILD "Batch NoorRay C++ sources for faster compilation" ON)
option(NR_ENABLE_PCH "Precompile common NoorRay C++ headers" ON)
option(NR_BUILD_PYTHON "Build the pynoorray Python bindings" OFF)
set(NR_UNITY_BUILD_BATCH_SIZE "8" CACHE STRING "Maximum sources per NoorRay unity batch")
set(NR_CUDA_SPLIT_COMPILE_JOBS "0" CACHE STRING "NVCC optimization threads per CUDA translation unit (0 = all cores)")
if(UNIX AND NOT APPLE)
    set(CMAKE_SKIP_BUILD_RPATH ON)
endif()
set(NR_CUDA_ARCH "89" CACHE STRING "CUDA architecture (e.g. 89 for sm_89)")
set(CMAKE_CUDA_ARCHITECTURES "${NR_CUDA_ARCH}")
# A single PTX file is generated below, so select the first architecture when
# the build is configured with multiple CUDA architectures.
list(GET NR_CUDA_ARCH 0 NR_OPTIX_CUDA_ARCH)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMAKE")

find_package(CUDAToolkit 13 REQUIRED)
if(NOT CUDAToolkit_VERSION_MAJOR EQUAL 13)
    message(FATAL_ERROR "NoorRay requires CUDA Toolkit 13.x")
endif()
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
set(SDL_ALSA OFF CACHE BOOL "NoorRay does not use SDL audio" FORCE)
add_subdirectory(external/SDL)
if(TARGET SDL3-static)
    set_target_properties(SDL3-static PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

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

# TinyEXR + miniz (32-bit floating-point OpenEXR output)
add_library(tinyexr STATIC
        external/tinyexr/tinyexr.cc
        external/tinyexr/deps/miniz/miniz.c)
target_include_directories(tinyexr PUBLIC
        ${PROJECT_SOURCE_DIR}/external/tinyexr
        ${PROJECT_SOURCE_DIR}/external/tinyexr/deps/miniz)

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
set(TBBMALLOC_BUILD OFF CACHE BOOL "" FORCE)
add_subdirectory(external/tbb)

# ROSS camera-lens ray tracing (kept as an unmodified submodule).
include(cmake/ROSS.cmake)

# GaussForge Gaussian readers. Use only the import sources NoorRay needs so the
# upstream CLI and benchmark targets do not become part of the main build.
include(FetchContent)
include(external/GaussForge/cmakes/zlib.cmake)
if(TARGET zlibstatic)
    set_target_properties(zlibstatic PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

FetchContent_Declare(
        libwebp
        GIT_REPOSITORY https://github.com/webmproject/libwebp.git
        GIT_TAG v1.4.0
        GIT_SHALLOW TRUE
)
set(WEBP_BUILD_ANIM_UTILS OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_CWEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_DWEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_GIF2WEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_IMG2WEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_VWEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_WEBPINFO OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_LIBWEBPMUX OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_WEBPMUX OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(libwebp)

include(external/GaussForge/cmakes/spz.cmake)

add_library(gauss_forge_readers STATIC
        external/GaussForge/src/core/validate.cpp
        external/GaussForge/src/io/ply_auto.cpp
        external/GaussForge/src/io/ply_reader.cpp
        external/GaussForge/src/io/ply_writer.cpp
        external/GaussForge/src/io/ply_compressed_reader.cpp
        external/GaussForge/src/io/splat_reader.cpp
        external/GaussForge/src/io/ksplat_reader.cpp
        external/GaussForge/src/io/spz_reader.cpp
        external/GaussForge/src/io/sog_reader.cpp
        external/GaussForge/src/io/sog_writer.cpp
)
set_target_properties(gauss_forge_readers PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gauss_forge_readers PUBLIC
        ${PROJECT_SOURCE_DIR}/external/GaussForge/include
)
target_link_libraries(gauss_forge_readers PUBLIC spz::spz webp nlohmann_json::nlohmann_json ZLIB::ZLIB)

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

find_program(NR_SLANGC_EXECUTABLE
        NAMES slangc
        HINTS
        "${VULKAN_SDK_ROOT}/bin"
        "$ENV{VULKAN_SDK}/bin"
        NO_CACHE
        REQUIRED
)

set(NR_CORE_DIR "${PROJECT_SOURCE_DIR}/src/libnoorray")
set(NR_APP_DIR "${PROJECT_SOURCE_DIR}/src/app")
set(SHADER_DIR     "${NR_CORE_DIR}/Shaders")

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
        "${SHADER_DIR}/Viewport/Viewport.spv"
        "${SHADER_DIR}/Viewport/ViewportBillboards.spv"
)

add_custom_command(
        OUTPUT ${SHADER_OUTPUTS}
        COMMAND "${NR_SLANGC_EXECUTABLE}" "${SHADER_DIR}/Viewport/Viewport.slang" -entry main ${SLANGC_BASE} -o "${SHADER_DIR}/Viewport/Viewport.spv"
        COMMAND "${NR_SLANGC_EXECUTABLE}" "${SHADER_DIR}/Viewport/ViewportBillboards.slang" -entry vertMain -stage vertex -entry fragMain -stage fragment ${SLANGC_BASE} -o "${SHADER_DIR}/Viewport/ViewportBillboards.spv"
        DEPENDS
        "${SHADER_DIR}/Viewport/Viewport.slang"
        "${SHADER_DIR}/Viewport/ViewportBillboards.slang"
        COMMENT "Compiling the Vulkan viewport shaders"
        VERBATIM
)

add_custom_target(NoorRayShaders DEPENDS ${SHADER_OUTPUTS})

# Generate embed wrapper headers with absolute paths (GCC 15 #embed only resolves
# absolute or source-relative paths, not -I search paths)
cmake_path(SET NR_GEN "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${NR_GEN}")

file(CONFIGURE OUTPUT "${NR_GEN}/ViewportSpv.h" CONTENT
"#pragma once
static constexpr unsigned char noorRayViewportSpv[] = {
    #embed \"${SHADER_DIR}/Viewport/Viewport.spv\"
};
static constexpr size_t noorRayViewportSpvLength = sizeof(noorRayViewportSpv);\n" @ONLY)

file(CONFIGURE OUTPUT "${NR_GEN}/ViewportBillboardsSpv.h" CONTENT
"#pragma once
static constexpr unsigned char noorRayViewportBillboardsSpv[] = {
    #embed \"${SHADER_DIR}/Viewport/ViewportBillboards.spv\"
};
static constexpr size_t noorRayViewportBillboardsSpvLength = sizeof(noorRayViewportBillboardsSpv);\n" @ONLY)

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
set_target_properties(imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(imgui PUBLIC
        ${PROJECT_SOURCE_DIR}/external/imgui
        ${PROJECT_SOURCE_DIR}/external/ImGuizmo
        ${PROJECT_SOURCE_DIR}/external/SDL/include
        ${Vulkan_INCLUDE_DIR}
)
target_link_libraries(imgui PUBLIC
        Vulkan::Vulkan
        SDL3::SDL3-static
)

add_library(NoorRayUiSupport STATIC
        src/app/UI/ImGuiManager.cpp
        src/app/UI/ImGuiManager.h
        src/app/UI/ImGuiComponent.h)
set_target_properties(NoorRayUiSupport PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(NoorRayUiSupport PUBLIC cxx_std_23)
target_include_directories(NoorRayUiSupport PUBLIC
        "${NR_APP_DIR}"
        "${NR_CORE_DIR}"
        "${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include"
        "${CUDAToolkit_INCLUDE_DIRS}"
        "${OPTIX_INCLUDE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(NoorRayUiSupport PUBLIC
        imgui
        Vulkan::Vulkan
        SDL3::SDL3-static
        glm)

add_library(NoorRayIO STATIC
        src/libnoorray/IO/BitmapReader.cpp
        src/libnoorray/IO/BitmapWriter.cpp
        src/libnoorray/StbImpl.cpp)
set_target_properties(NoorRayIO PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(NoorRayIO PUBLIC cxx_std_23)
target_include_directories(NoorRayIO PUBLIC "${NR_CORE_DIR}")
target_link_libraries(NoorRayIO PUBLIC glm stb tinyexr)

set(NR_OPTIX_DEPFILE "${CMAKE_CURRENT_BINARY_DIR}/generated/NoorRayOptix.d")

add_custom_command(
        OUTPUT "${NR_OPTIX_PTX}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_CUDA_COMPILER}"
                -ccbin "${CMAKE_CXX_COMPILER}"
                --ptx --gpu-architecture=compute_${NR_OPTIX_CUDA_ARCH} --std=c++20
                --generate-line-info --expt-relaxed-constexpr
                "--split-compile=${NR_CUDA_SPLIT_COMPILE_JOBS}"
                --generate-nonsystem-dependencies-with-compile
                --dependency-output "${NR_OPTIX_DEPFILE}"
                -DNR_BACKEND_CUDA=1 -DNR_OPTIX_PTX_BUILD=1
                "-I${NR_CORE_DIR}"
                "-I${NR_APP_DIR}"
                "-I${PROJECT_SOURCE_DIR}/external/glm"
                "-I${OPTIX_INCLUDE_DIR}"
                -o "${NR_OPTIX_PTX}"
                "${NR_CORE_DIR}/Kernels/Extend.cu"
        DEPENDS
                src/libnoorray/Kernels/Extend.cu
                src/libnoorray/Kernels/Connect.cu
                src/libnoorray/Kernels/GaussianHit.cu
                src/libnoorray/Kernels/GaussianProxyOverdraw.cu
        DEPFILE "${NR_OPTIX_DEPFILE}"
        VERBATIM)
add_custom_target(NoorRayOptixIr DEPENDS "${NR_OPTIX_PTX}")

add_library(NoorRayCuda STATIC
        src/libnoorray/Raytracing/Blas.cpp
        src/libnoorray/Raytracing/GaussianProxyBlas.cpp
        src/libnoorray/CUDA/Unique/Texture.cpp
        src/libnoorray/CUDA/Unique/SharedImage.cpp
        src/libnoorray/CUDA/Unique/SharedBuffer.cpp
        src/libnoorray/CUDA/rstd/Memory.cpp
        src/libnoorray/Mesh/OpenPbrEnergy.cpp
        src/libnoorray/Kernels/Generate.cu
        src/libnoorray/Kernels/Shade.cu
        src/libnoorray/Kernels/Finalize.cu
        src/libnoorray/Kernels/GenerateAov.cu
        src/libnoorray/Kernels/ShadeAov.cu
        src/libnoorray/Raytracing/Tlas.cpp
        src/libnoorray/Raytracing/RgbSpectrumTableData.cpp
        src/libnoorray/Raytracing/Raytracer.cpp
        src/libnoorray/Training/GaussianTrainRenderer.cpp
        src/libnoorray/Training/GaussianTrainer.cu)
add_dependencies(NoorRayCuda NoorRayOptixIr)
set_target_properties(NoorRayCuda PROPERTIES CUDA_ARCHITECTURES "${NR_CUDA_ARCH}")
set_target_properties(NoorRayCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
        POSITION_INDEPENDENT_CODE ON)
target_compile_features(NoorRayCuda PUBLIC cxx_std_20 cuda_std_20)
target_compile_definitions(NoorRayCuda PUBLIC
        NR_BACKEND_CUDA=1
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
target_include_directories(NoorRayCuda PUBLIC
        "${NR_CORE_DIR}"
        "${NR_APP_DIR}"
        "${PROJECT_SOURCE_DIR}/external/glm"
        "${PROJECT_SOURCE_DIR}/external/SDL/include"
        "${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include"
        "${PROJECT_SOURCE_DIR}/external/portable-file-dialogs"
        "${Vulkan_INCLUDE_DIR}"
        "${OPTIX_INCLUDE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_options(NoorRayCuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info;--extended-lambda;--expt-relaxed-constexpr;--split-compile=${NR_CUDA_SPLIT_COMPILE_JOBS}>)
target_link_libraries(NoorRayCuda PUBLIC
        CUDA::cudart_static
        CUDA::cuda_driver
        glm
        stb
        Vulkan::Vulkan
        SDL3::SDL3-static
        portable_file_dialogs
        GPUOpen::VulkanMemoryAllocator
        gauss_forge_readers)
target_link_libraries(NoorRayCuda PUBLIC libross)

file(GLOB_RECURSE NR_CORE_FILES CONFIGURE_DEPENDS
        "${NR_CORE_DIR}/*.cpp"
        "${NR_CORE_DIR}/*.h")
list(FILTER NR_CORE_FILES EXCLUDE REGEX "/CUDA/")
list(FILTER NR_CORE_FILES EXCLUDE REGEX "/Kernels/")
list(FILTER NR_CORE_FILES EXCLUDE REGEX "/Raytracing/Raytracer\\.cpp$")
list(FILTER NR_CORE_FILES EXCLUDE REGEX "/Raytracing/RgbSpectrumTableData\\.cpp$")
list(FILTER NR_CORE_FILES EXCLUDE REGEX "/Raytracing/Blas\\.cpp$")
list(FILTER NR_CORE_FILES EXCLUDE REGEX "/Raytracing/GaussianProxyBlas\\.cpp$")
list(FILTER NR_CORE_FILES EXCLUDE REGEX "/Raytracing/Tlas\\.cpp$")
list(FILTER NR_CORE_FILES EXCLUDE REGEX "/Mesh/OpenPbrEnergy\\.cpp$")
list(FILTER NR_CORE_FILES EXCLUDE REGEX "/IO/BitmapReader\\.cpp$")
list(FILTER NR_CORE_FILES EXCLUDE REGEX "/IO/BitmapWriter\\.cpp$")
list(FILTER NR_CORE_FILES EXCLUDE REGEX "/StbImpl\\.cpp$")

add_library(libnoorray STATIC ${NR_CORE_FILES})
set_target_properties(libnoorray PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
        POSITION_INDEPENDENT_CODE ON)
target_compile_features(libnoorray PUBLIC cxx_std_23)
target_include_directories(libnoorray
        PUBLIC
        "${NR_CORE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
        PRIVATE
        "${NR_APP_DIR}")
target_compile_definitions(libnoorray PUBLIC
        NR_BACKEND_CUDA=1
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        NOORRAY_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets"
        NR_ROSS_RESOURCE_DIR="${PROJECT_SOURCE_DIR}/external/ROSS/resources")
target_link_libraries(libnoorray
        PUBLIC
        NoorRayCuda
        NoorRayIO
        portable_file_dialogs
        glm
        Vulkan::Vulkan
        SDL3::SDL3-static
        tinygltf
        tinyobjloader
        tinyexr
        stb
        gauss_forge_readers
        GPUOpen::VulkanMemoryAllocator
        glaze
        TBB::tbb
        PRIVATE
        imgui
        NoorRayUiSupport)

file(GLOB_RECURSE NR_APP_FILES CONFIGURE_DEPENDS
        "${NR_APP_DIR}/*.cpp"
        "${NR_APP_DIR}/*.h")
list(FILTER NR_APP_FILES EXCLUDE REGEX "/UI/ImGuiManager\\.cpp$")

if(NR_BUILD_PYTHON)
    if(CMAKE_VERSION VERSION_LESS 3.18)
        set(NR_PYTHON_DEV_MODULE Development)
    else()
        set(NR_PYTHON_DEV_MODULE Development.Module)
    endif()

    find_package(Python 3.12 COMPONENTS Interpreter ${NR_PYTHON_DEV_MODULE} REQUIRED)
    add_subdirectory("${PROJECT_SOURCE_DIR}/external/nanobind" "${CMAKE_BINARY_DIR}/pynoorray/nanobind" EXCLUDE_FROM_ALL)

    nanobind_add_module(_pynoorray
            src/pynoorray/PyNoorRay.cpp
            src/pynoorray/PyRenderSession.cpp
            src/pynoorray/PyRenderSession.h
            src/pynoorray/__init__.py)
    set_target_properties(_pynoorray PROPERTIES DEBUG_POSTFIX "")
    target_compile_features(_pynoorray PRIVATE cxx_std_23)
    target_include_directories(_pynoorray PRIVATE
            "${PROJECT_SOURCE_DIR}/src/pynoorray"
            "${NR_CORE_DIR}"
            "${NR_APP_DIR}")
    target_link_libraries(_pynoorray PRIVATE
            "$<LINK_GROUP:RESCAN,libnoorray,NoorRayCuda,NoorRayIO,libross>"
            NoorRayUiSupport
            CUDA::cudart_static)
    target_compile_definitions(_pynoorray PRIVATE
            NR_BACKEND_CUDA=1
            VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
    # Keep the source layout flat. The extension and its Python files are
    # installed into one package directory next to each other.
    add_custom_command(TARGET _pynoorray POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_FILE_DIR:_pynoorray>/pynoorray"
            COMMAND ${CMAKE_COMMAND} -E rm -f
                    "$<TARGET_FILE_DIR:_pynoorray>/pynoorray/training.py"
                    "$<TARGET_FILE_DIR:_pynoorray>/pynoorray/colmap.py"
            COMMAND ${CMAKE_COMMAND} -E copy
                    "${PROJECT_SOURCE_DIR}/src/pynoorray/__init__.py"
                    "$<TARGET_FILE_DIR:_pynoorray>/pynoorray/__init__.py"
            COMMAND ${CMAKE_COMMAND} -E copy
                    "$<TARGET_FILE:_pynoorray>"
                    "$<TARGET_FILE_DIR:_pynoorray>/pynoorray/"
            COMMENT "Copying pynoorray package")
endif()

# Resource handling on Windows
if(WIN32)
    set(RESOURCE_RC_PATH "${CMAKE_BINARY_DIR}/resource.rc")
    file(WRITE "${RESOURCE_RC_PATH}"
            "MAINICON ICON \"${PROJECT_SOURCE_DIR}/assets/icons/icon.ico\"\n"
    )
    message(STATUS "Generated resource.rc at ${RESOURCE_RC_PATH}")
    list(APPEND NR_APP_FILES "${RESOURCE_RC_PATH}")
endif()

# Executable target — WIN32 only applies on Windows (suppresses the console window
# for GUI mode), but CLI mode still works because SDL_MAIN_HANDLED is set.
if(WIN32)
    add_executable(NoorRay WIN32 ${NR_APP_FILES})
else()
    add_executable(NoorRay ${NR_APP_FILES})
endif()
set_target_properties(NoorRay PROPERTIES PREFIX "")
set_target_properties(NoorRay PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_dependencies(NoorRay NoorRayShaders)

target_include_directories(NoorRay PRIVATE
        ${NR_APP_DIR}
        ${NR_CORE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/generated
)

# Link against all libraries
target_link_libraries(NoorRay PRIVATE
        libnoorray
        NoorRayUiSupport
        NoorRayCuda
        libross
        portable_file_dialogs
        glm
        imgui
        Vulkan::Vulkan
        SDL3::SDL3-static
        tinygltf
        tinyobjloader
        tinyexr
        stb
        gauss_forge_readers
        ImViewGuizmo
        GPUOpen::VulkanMemoryAllocator
        glaze
        TBB::tbb
)

target_compile_definitions(NoorRay PRIVATE
        SDL_MAIN_HANDLED
        NR_BACKEND_CUDA=1)
target_compile_definitions(NoorRay PRIVATE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

if(NR_ENABLE_UNITY_BUILD)
    set_target_properties(NoorRay PROPERTIES
            UNITY_BUILD ON
            UNITY_BUILD_BATCH_SIZE "${NR_UNITY_BUILD_BATCH_SIZE}")
    set_source_files_properties(
        src/app/NoorRayApp.cpp
        src/libnoorray/Scene/SceneImporter.cpp
        src/libnoorray/StbImpl.cpp
        src/app/UI/ImGuiManager.cpp
        src/libnoorray/Vulkan/Viewport.cpp
        src/libnoorray/Vulkan/VmaImpl.cpp
        src/libnoorray/Mesh/GaussianAsset.cpp
        src/libnoorray/Raytracing/GaussianProxyBlas.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(NR_ENABLE_PCH)
    target_precompile_headers(NoorRay PRIVATE
            <algorithm>
            <array>
            <cstddef>
            <cstdint>
            <memory>
            <string>
            <utility>
            <vector>
            <glm/glm.hpp>
            <vulkan/vulkan.hpp>)
endif()

if(MSVC)
    set_property(TARGET NoorRay PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
elseif(UNIX AND NOT APPLE)
    target_link_options(NoorRay PRIVATE -static-libgcc -static-libstdc++)
endif()

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

option(BUILD_TESTING "Build and register end-to-end tests" OFF)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
