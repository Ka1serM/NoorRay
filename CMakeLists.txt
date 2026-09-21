cmake_minimum_required(VERSION 3.25)

if(NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "Set CMAKE_CXX_COMPILER")
endif()
project(NoorRay VERSION 0.1.0 LANGUAGES C CXX)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "NoorRay requires Linux")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Directory-scoped, so a host project keeps its own setting.
set(BUILD_SHARED_LIBS OFF)

# NoorRay can be the top-level project or be added with add_subdirectory()
# by a host application, which links the NoorRay::noorray target. Build-tree
# layout and build type are the host's to decide in that case.
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_SKIP_BUILD_RPATH ON)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(NR_CORE_DIR "${PROJECT_SOURCE_DIR}/src/libnoorray")
set(NR_APP_DIR "${PROJECT_SOURCE_DIR}/src/app")

# USD provides scene import/export only. Hosts that just drive the renderer -
# an embedded viewport, for instance - never reach that path, and USD is by far
# the heaviest thing in the dependency graph: it pulls a Blender-provided
# libusd_ms.so and libpython into every link. Keeping it optional lets those
# hosts build and ship without either.
# Keep the default editor/renderer build small. USD, Python, tests, and the
# standalone API examples remain opt-in integrations for developers who need
# them; they should not enlarge the normal NoorRay dependency graph.
option(NR_ENABLE_USD "Build USD scene import/export" OFF)
# The editor (and the tests of its UI) are what need SDL, Dear ImGui and the
# file dialogs; the library needs none of them. A host that embeds the
# renderer turns the editor off and brings its own copies of those targets.
option(NR_BUILD_APPLICATION "Build the standalone NoorRay application" ${PROJECT_IS_TOP_LEVEL})
option(NR_BUILD_TESTS "Build tests" OFF)
if(NR_BUILD_APPLICATION OR NR_BUILD_TESTS)
    set(NR_BUILD_EDITOR_DEPENDENCIES ON)
else()
    set(NR_BUILD_EDITOR_DEPENDENCIES OFF)
endif()
# Consumed by external/NoorRHI, which owns the example targets, so it must be declared
# before that subdirectory is added.
option(NR_BUILD_EXAMPLES "Build the standalone NoorRHI API examples" OFF)
if(NR_ENABLE_USD)
    set(NR_USD_ROOT "${PROJECT_SOURCE_DIR}/external/blender-usd" CACHE PATH
        "Blender USD SDK root (auto-vendored by default)")
    include(cmake/NoorRayUSD.cmake)
endif()

function(nr_header_only target dir)
    add_library(${target} INTERFACE)
    target_include_directories(${target} INTERFACE "${dir}")
endfunction()

# ---------------------------------------------------------------------------
# SDL3, glm, PFD, VMA
# ---------------------------------------------------------------------------
if(NR_BUILD_EDITOR_DEPENDENCIES)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    add_subdirectory(external/SDL)
    if(TARGET SDL3-static)
        set_target_properties(SDL3-static PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
    add_subdirectory(external/portable-file-dialogs)
endif()
if(NOT TARGET glm)
    add_subdirectory(external/glm)
endif()
if(NOT TARGET GPUOpen::VulkanMemoryAllocator)
    add_subdirectory(external/VulkanMemoryAllocator)
endif()

# ---------------------------------------------------------------------------
# Header-only libraries
# ---------------------------------------------------------------------------
nr_header_only(tinygltf     "${PROJECT_SOURCE_DIR}/external/tinygltf")
nr_header_only(tinyobjloader "${PROJECT_SOURCE_DIR}/external/tinyobjloader")
nr_header_only(stb          "${PROJECT_SOURCE_DIR}/external/stb")
if(NR_BUILD_EDITOR_DEPENDENCIES)
    nr_header_only(ImViewGuizmo "${PROJECT_SOURCE_DIR}/external/ImViewGuizmo")
endif()
nr_header_only(glaze        "${PROJECT_SOURCE_DIR}/external/glaze/include")

# ---------------------------------------------------------------------------
# tinyply
# ---------------------------------------------------------------------------
add_library(tinyply STATIC external/tinyply/source/tinyply.cpp)
set_target_properties(tinyply PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tinyply PUBLIC "${PROJECT_SOURCE_DIR}/external/tinyply/source")

# ---------------------------------------------------------------------------
# TinyEXR
# ---------------------------------------------------------------------------
add_library(tinyexr STATIC
    external/tinyexr/tinyexr.cc
    external/tinyexr/deps/miniz/miniz.c)
set_target_properties(tinyexr PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tinyexr PUBLIC
    "${PROJECT_SOURCE_DIR}/external/tinyexr"
    "${PROJECT_SOURCE_DIR}/external/tinyexr/deps/miniz")

# ---------------------------------------------------------------------------
# TBB
# ---------------------------------------------------------------------------
set(TBB_TEST OFF CACHE BOOL "" FORCE)
set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TBB_STRICT OFF CACHE BOOL "" FORCE)
set(TBBMALLOC_BUILD OFF CACHE BOOL "" FORCE)
add_subdirectory(external/tbb)


# ---------------------------------------------------------------------------
# GaussForge (gaussian splat readers)
# ---------------------------------------------------------------------------
include(FetchContent)
include(external/GaussForge/cmakes/zlib.cmake)
if(TARGET zlibstatic)
    set_target_properties(zlibstatic PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
include(external/GaussForge/cmakes/sog.cmake)
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
    external/GaussForge/src/io/sog_reader.cpp)
set_target_properties(gauss_forge_readers PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gauss_forge_readers PUBLIC "${PROJECT_SOURCE_DIR}/external/GaussForge/include")
target_link_libraries(gauss_forge_readers PUBLIC spz::spz webp nlohmann_json::nlohmann_json ZLIB::ZLIB)

# ---------------------------------------------------------------------------
# Vulkan + slangc
# ---------------------------------------------------------------------------
find_package(Vulkan REQUIRED)
# An unset VULKAN_SDK must not turn into a "/bin" hint: that silently picks the
# distribution's older tools over the SDK's (the system spirv-val takes tens of
# minutes on the ray-tracing modules).
set(NR_VULKAN_SDK_BIN_HINTS "/home/marcel/Programs/VulkanSDK/current/bin")
if(DEFINED ENV{VULKAN_SDK} AND NOT "$ENV{VULKAN_SDK}" STREQUAL "")
    list(PREPEND NR_VULKAN_SDK_BIN_HINTS "$ENV{VULKAN_SDK}/bin")
endif()
find_program(NR_SLANGC_EXECUTABLE NAMES slangc
    HINTS ${NR_VULKAN_SDK_BIN_HINTS}
    NO_CACHE REQUIRED)
# Optional: when present, every compiled module is checked against the Vulkan
# 1.4 SPIR-V environment, which is where descriptor-heap lowering bugs show up.
find_program(NR_SPIRV_VAL_EXECUTABLE NAMES spirv-val
    HINTS ${NR_VULKAN_SDK_BIN_HINTS}
    NO_CACHE)
set(Vulkan_INCLUDE_DIR "${Vulkan_INCLUDE_DIR}" CACHE PATH "")

# ---------------------------------------------------------------------------
# ImGui + ImGuizmo
# ---------------------------------------------------------------------------
if(NR_BUILD_EDITOR_DEPENDENCIES)
    add_library(imgui STATIC
        external/imgui/imgui.cpp
        external/imgui/imgui_draw.cpp
        external/imgui/imgui_tables.cpp
        external/imgui/imgui_widgets.cpp
        external/imgui/backends/imgui_impl_vulkan.cpp
        external/imgui/backends/imgui_impl_sdl3.cpp
        external/ImGuizmo/ImGuizmo.cpp)
    set_target_properties(imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(imgui PUBLIC
        "${PROJECT_SOURCE_DIR}/external/imgui"
        "${PROJECT_SOURCE_DIR}/external/ImGuizmo"
        "${PROJECT_SOURCE_DIR}/external/SDL/include")
    target_link_libraries(imgui PUBLIC Vulkan::Vulkan SDL3::SDL3-static)

    # ImNodeFlow node editor with native zoom, links and customizable nodes.
    add_library(imnodeflow STATIC external/ImNodeFlow/src/ImNodeFlow.cpp)
    target_include_directories(imnodeflow PUBLIC
        "${PROJECT_SOURCE_DIR}/external/ImNodeFlow/include"
        "${PROJECT_SOURCE_DIR}/external/ImNodeFlow/src")
    target_compile_definitions(imnodeflow PUBLIC IMGUI_DEFINE_MATH_OPERATORS)
    target_link_libraries(imnodeflow PUBLIC imgui)
endif()

# ---------------------------------------------------------------------------
# NoorRayIO (image I/O, stb wrapper)
# ---------------------------------------------------------------------------
add_library(NoorRayIO STATIC
    src/libnoorray/IO/BitmapReader.cpp
    src/libnoorray/IO/BitmapWriter.cpp
    src/libnoorray/IO/StbImpl.cpp)
set_target_properties(NoorRayIO PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(NoorRayIO PUBLIC cxx_std_23)
target_include_directories(NoorRayIO PUBLIC "${NR_CORE_DIR}")
target_link_libraries(NoorRayIO PUBLIC glm stb tinyexr)

# ---------------------------------------------------------------------------
# NoorRHI - small Vulkan graphics abstraction
# ---------------------------------------------------------------------------
# NoorRHI is its own repository (external/NoorRHI): it compiles its own VMA
# implementation TU, builds, tests and installs standalone. It reuses the
# GPUOpen::VulkanMemoryAllocator target defined above instead of finding its
# own. Its examples follow NoorRay's switch; its tests run from its own build.
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/NoorRHI/CMakeLists.txt")
    message(FATAL_ERROR "external/NoorRHI is missing. Link or check out the NoorRHI repository there.")
endif()
set(NOORRHI_BUILD_EXAMPLES ${NR_BUILD_EXAMPLES} CACHE BOOL "" FORCE)
set(NOORRHI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(external/NoorRHI)

# ---------------------------------------------------------------------------
# NRD (NVIDIA Real-time Denoisers) for the realtime renderer
# ---------------------------------------------------------------------------
# Built as a static library embedding its SPIR-V. NoorRay dispatches the NRD
# pipelines itself (Realtime/Denoiser.cpp), so NRI is not pulled in.
# Normal encoding 2 (R10G10B10A2) packs [0, 1] values, which float textures hold.
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/NRD/CMakeLists.txt")
    message(FATAL_ERROR "external/NRD is empty. Run: git submodule update --init external/NRD")
endif()
set(NRD_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
set(NRD_NRI OFF CACHE BOOL "" FORCE)
set(NRD_EMBEDS_SPIRV_SHADERS ON CACHE BOOL "" FORCE)
set(NRD_EMBEDS_DXIL_SHADERS OFF CACHE BOOL "" FORCE)
set(NRD_EMBEDS_DXBC_SHADERS OFF CACHE BOOL "" FORCE)
set(NRD_SUPPORTS_QUAD_INTRINSICS OFF CACHE BOOL "" FORCE)
set(NRD_NORMAL_ENCODING "2" CACHE STRING "" FORCE)
set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "" FORCE)
set(NRD_SHADERS_PATH "${PROJECT_BINARY_DIR}/nrd_shaders" CACHE STRING "" FORCE)
add_subdirectory(external/NRD)
set_target_properties(NRD PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# RTXDI (NVIDIA ReSTIR DI / GI and ReGIR) for the realtime renderer
# ---------------------------------------------------------------------------
# The host library only computes parameters and buffer layouts; its HLSL
# resampling headers are compiled into the realtime Slang passes, which
# supply RTXDI's application bridge (RtxdiBridge.slang).
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/RTXDI-Library/CMakeLists.txt")
    message(FATAL_ERROR "external/RTXDI-Library is empty. Run: git submodule update --init external/RTXDI-Library")
endif()
add_subdirectory(external/RTXDI-Library)
set_target_properties(Rtxdi PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# FSR 3.1 upscaling (AMD FidelityFX SDK) for the realtime renderer
# ---------------------------------------------------------------------------
# external/FidelityFX-SDK is pinned to v1.1.4 (FSR 3.1.4), the last SDK with a
# Vulkan backend. Its host code, FSR component and Vulkan backend are built
# here with Realtime/FfxPlatform.h force-included for the MSVC functions they
# use. The SDK's shader compiler is Windows-only, so the GLSL passes are
# compiled with glslang below and reflected at runtime (FsrShaderBlobs.cpp).
set(NR_FFX_SDK_DIR "${PROJECT_SOURCE_DIR}/external/FidelityFX-SDK/sdk")
if(NOT EXISTS "${NR_FFX_SDK_DIR}/src/backends/vk/ffx_vk.cpp")
    message(FATAL_ERROR "external/FidelityFX-SDK is missing or not at v1.1.4. Run: git submodule update --init external/FidelityFX-SDK")
endif()
add_library(NoorRayFsr STATIC
    "${NR_FFX_SDK_DIR}/src/backends/vk/ffx_vk.cpp"
    "${NR_FFX_SDK_DIR}/src/components/fsr3upscaler/ffx_fsr3upscaler.cpp"
    "${NR_FFX_SDK_DIR}/src/shared/ffx_assert.cpp"
    "${NR_FFX_SDK_DIR}/src/shared/ffx_message.cpp"
    "${NR_FFX_SDK_DIR}/src/shared/ffx_object_management.cpp"
    "${NR_FFX_SDK_DIR}/src/shared/ffx_breadcrumbs_list.cpp"
    "${NR_FFX_SDK_DIR}/tools/ffx_shader_compiler/libs/SPIRV-Reflect/spirv_reflect.c"
    "${NR_CORE_DIR}/Realtime/FsrShaderBlobs.cpp")
set_target_properties(NoorRayFsr PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(NoorRayFsr PUBLIC cxx_std_23)
target_include_directories(NoorRayFsr SYSTEM PUBLIC "${NR_FFX_SDK_DIR}/include")
target_include_directories(NoorRayFsr SYSTEM PRIVATE
    "${NR_FFX_SDK_DIR}/src/shared"
    "${NR_FFX_SDK_DIR}/src/backends/shared"
    "${NR_FFX_SDK_DIR}/src/components"
    "${NR_FFX_SDK_DIR}/src/components/fsr3upscaler"
    "${NR_FFX_SDK_DIR}/tools/ffx_shader_compiler/libs/SPIRV-Reflect")
target_include_directories(NoorRayFsr PRIVATE "${NR_CORE_DIR}")
target_compile_definitions(NoorRayFsr PUBLIC FFX_GCC FFX_FSR3UPSCALER)
# The SDK is third-party code written for MSVC: keep its warnings out of the
# build. #embed is a C23 extension in C++ for Clang.
target_compile_options(NoorRayFsr PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-include;${NR_CORE_DIR}/Realtime/FfxPlatform.h>"
    "$<$<COMPILE_LANGUAGE:CXX>:-w>"
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:-Wno-c23-extensions>")
target_link_libraries(NoorRayFsr PUBLIC Vulkan::Vulkan)
# The Vulkan backend places its bookkeeping structures in one scratch arena at
# 4-byte offsets. Optimized GCC code writes those structures with aligned SSE
# moves, which fault there. The backend only schedules and records FSR's
# dispatches, so it is built without optimization; everything else is not.
set_source_files_properties("${NR_FFX_SDK_DIR}/src/backends/vk/ffx_vk.cpp"
    PROPERTIES COMPILE_OPTIONS "-O0")

# ---------------------------------------------------------------------------
# libnoorray (core engine library)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE NR_CORE_FILES CONFIGURE_DEPENDS
    "${NR_CORE_DIR}/*.cpp" "${NR_CORE_DIR}/*.h")
set(NR_CORE_X
    "/Materials/MaterialX/MaterialXDocument\\.cpp$" "/Materials/MaterialX/SlangMaterial[A-Za-z]*\\.cpp$"
    "/IO/BitmapReader\\.cpp$" "/IO/BitmapWriter\\.cpp$" "/StbImpl\\.cpp$"
    "/Materials/SVM/references/"
    "/Realtime/FsrShaderBlobs\\.cpp$")
if(NOT NR_ENABLE_USD)
    # The stub beside it implements the same three entry points, so callers in
    # SceneReader/SceneWriter/SceneImporter need no conditional compilation.
    list(APPEND NR_CORE_X "/Scene/Import/SceneUsd\\.cpp$")
else()
    list(APPEND NR_CORE_X "/Scene/Import/SceneUsdDisabled\\.cpp$")
endif()
foreach(p IN LISTS NR_CORE_X)
    list(FILTER NR_CORE_FILES EXCLUDE REGEX "${p}")
endforeach()
add_library(libnoorray STATIC ${NR_CORE_FILES})
add_library(NoorRay::noorray ALIAS libnoorray)
add_dependencies(libnoorray NoorRayShaders)
set_target_properties(libnoorray PROPERTIES
    OUTPUT_NAME noorray
    POSITION_INDEPENDENT_CODE ON)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(libnoorray PRIVATE -Wno-c23-extensions)
endif()
target_compile_features(libnoorray PUBLIC cxx_std_23)
target_include_directories(libnoorray PUBLIC "${NR_CORE_DIR}")
target_compile_definitions(libnoorray PUBLIC
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
# Public: what libnoorray's own headers include. Everything else is an
# implementation detail; as a static library it still reaches the link line.
target_link_libraries(libnoorray
    PUBLIC
        NoorRHI::NoorRHI glm Vulkan::Vulkan MaterialXCore
    PRIVATE
        NoorRayIO tinygltf tinyobjloader tinyply tinyexr stb gauss_forge_readers
        GPUOpen::VulkanMemoryAllocator glaze TBB::tbb
        NoorRayMaterialX MaterialXFormat NRD Rtxdi NoorRayFsr)
if(NR_ENABLE_USD)
    target_link_libraries(libnoorray PUBLIC NoorRayUsd)
endif()

# ---------------------------------------------------------------------------
# MaterialX
# ---------------------------------------------------------------------------
set(NR_MATERIALX_PINNED_VERSION "1.39.5")
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/materialx/CMakeLists.txt")
    message(FATAL_ERROR "external/materialx is empty. Run: git submodule update --init --recursive external/materialx")
endif()
foreach(_v MATERIALX_BUILD_SHARED_LIBS MATERIALX_BUILD_GEN_OSL MATERIALX_BUILD_GEN_GLSL
         MATERIALX_BUILD_GEN_MSL MATERIALX_BUILD_PYTHON MATERIALX_BUILD_VIEWER
         MATERIALX_BUILD_GRAPH_EDITOR MATERIALX_BUILD_TESTS MATERIALX_TEST_RENDER
         MATERIALX_INSTALL_PYTHON)
    set(${_v} OFF CACHE BOOL "" FORCE)
endforeach()
set(MATERIALX_BUILD_GEN_MDL OFF CACHE BOOL "" FORCE)
# MaterialX 1.39.5 provides the Slang source generator. Keep other backends
# disabled: NoorRay consumes emitted Slang through its Vulkan RT ABI.
set(MATERIALX_BUILD_GEN_SLANG ON CACHE BOOL "" FORCE)
add_subdirectory(external/materialx)
foreach(t MaterialXCore MaterialXFormat MaterialXGenShader MaterialXGenHw MaterialXGenSlang)
    if(TARGET ${t})
        set_target_properties(${t} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
endforeach()
set(NR_MATERIALX_STDLIB_DIR "${PROJECT_SOURCE_DIR}/external/materialx/libraries")

add_library(NoorRayMaterialX STATIC
    src/libnoorray/Materials/MaterialX/MaterialXDocument.cpp
    src/libnoorray/Materials/MaterialX/SlangMaterialGenerator.cpp
    src/libnoorray/Materials/MaterialX/SlangMaterialCompiler.cpp)
target_compile_features(NoorRayMaterialX PUBLIC cxx_std_23)
target_include_directories(NoorRayMaterialX PUBLIC "${NR_CORE_DIR}")
# Generated materials are compiled at runtime with the Slang compiler library
# of the same SDK whose slangc builds the prebuilt shaders.
find_package(slang CONFIG REQUIRED
    HINTS $ENV{VULKAN_SDK}/lib/cmake/slang /home/marcel/Programs/VulkanSDK/current/lib/cmake/slang)
target_link_libraries(NoorRayMaterialX PUBLIC
    glm MaterialXGenSlang MaterialXGenShader MaterialXFormat MaterialXCore)
target_link_libraries(NoorRayMaterialX PRIVATE slang::slang)
# MaterialInterface.slang is compiled into every generated material.
target_compile_options(NoorRayMaterialX PRIVATE
    "--embed-dir=${PROJECT_SOURCE_DIR}/src/libnoorray/Shaders")
target_compile_definitions(NoorRayMaterialX PRIVATE
    NR_MATERIALX_STDLIB_DIR="${NR_MATERIALX_STDLIB_DIR}")
set_target_properties(NoorRayMaterialX PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Vulkan shaders (Viewport, ViewportBillboards)
# ---------------------------------------------------------------------------
set(NR_SHADER_DIR "${NR_CORE_DIR}/Shaders")
# Compiled SPIR-V is a build product: it lives in the binary tree and reaches
# the library through #embed, which searches this directory.
set(NR_SHADER_BUILD_DIR "${PROJECT_BINARY_DIR}/shaders")
target_compile_options(libnoorray PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:--embed-dir=${NR_SHADER_BUILD_DIR}>")
file(GLOB NR_RAYTRACER_SHADER_INCLUDES CONFIGURE_DEPENDS
    "${NR_SHADER_DIR}/Raytracer/*.slang")
file(GLOB NR_REALTIME_SHADER_INCLUDES CONFIGURE_DEPENDS
    "${NR_SHADER_DIR}/RealtimeRaytracer/*.slang")
# NVIDIA SHaRC radiance cache: HLSL headers included by the realtime shaders.
# SharcCommon.h leaves out-parameters unset on its early-return paths (41018).
set(NR_SHARC_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/SHARC/include")
if(NOT EXISTS "${NR_SHARC_INCLUDE_DIR}/SharcCommon.h")
    message(FATAL_ERROR "external/SHARC is empty. Run: git submodule update --init external/SHARC")
endif()
file(GLOB NR_SHARC_HEADERS CONFIGURE_DEPENDS "${NR_SHARC_INCLUDE_DIR}/*.h")
set(NR_SHARC_SHADER_FLAGS -I "${NR_SHARC_INCLUDE_DIR}" -Wno-41018)
# NRD front-end helpers (NRD.hlsli) packing the denoiser inputs. Its
# NRDConfig.hlsli is generated when external/NRD is configured below.
set(NR_NRD_SHADER_FLAGS -I "${PROJECT_SOURCE_DIR}/external/NRD/Shaders")
# RTXDI's resampling headers, included by the realtime shaders through
# RtxdiBridge.slang.
set(NR_RTXDI_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/RTXDI-Library/Include")
file(GLOB_RECURSE NR_RTXDI_HEADERS CONFIGURE_DEPENDS
    "${NR_RTXDI_INCLUDE_DIR}/Rtxdi/*.h" "${NR_RTXDI_INCLUDE_DIR}/Rtxdi/*.hlsli")
set(NR_RTXDI_SHADER_FLAGS -I "${NR_RTXDI_INCLUDE_DIR}")

# FSR 3.1 GLSL passes: one permutation, matching FsrUpscaler's context flags
# and the check in FsrShaderBlobs.cpp. The accumulate pass is also built with
# sharpening for FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN.
find_program(NR_GLSLANG_EXECUTABLE NAMES glslangValidator
    HINTS ${NR_VULKAN_SDK_BIN_HINTS} NO_CACHE REQUIRED)
set(NR_FSR_SHADER_SOURCE_DIR "${NR_FFX_SDK_DIR}/src/backends/vk/shaders/fsr3upscaler")
set(NR_FSR_GPU_DIR "${NR_FFX_SDK_DIR}/include/FidelityFX/gpu")
set(NR_FSR_SHADER_BUILD_DIR "${NR_SHADER_BUILD_DIR}/Fsr")
set(NR_FSR_SHADER_DEFINES
    -DFFX_GPU=1 -DFFX_GLSL=1
    -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0
    -DFFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0
    -DFFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1
    -DFFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0
    -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2
    -DFFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE=0
    -DFFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT=1
    -DFFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS=1
    -DFFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS=0
    -DFFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH=1)
file(GLOB NR_FSR_SHADER_DEPENDENCIES CONFIGURE_DEPENDS
    "${NR_FSR_SHADER_SOURCE_DIR}/*.glsl" "${NR_FSR_GPU_DIR}/*.h"
    "${NR_FSR_GPU_DIR}/fsr3upscaler/*.h")
set(NR_FSR_SHADER_OUT "")
function(nr_fsr_shader pass sharpening output)
    set(spv "${NR_FSR_SHADER_BUILD_DIR}/${output}.spv")
    add_custom_command(OUTPUT "${spv}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NR_FSR_SHADER_BUILD_DIR}"
        COMMAND "${NR_GLSLANG_EXECUTABLE}" -V --target-env vulkan1.2 -S comp -Os
            ${NR_FSR_SHADER_DEFINES}
            -DFFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING=${sharpening}
            "-I${NR_FSR_GPU_DIR}" "-I${NR_FSR_GPU_DIR}/fsr3upscaler"
            "${NR_FSR_SHADER_SOURCE_DIR}/ffx_fsr3upscaler_${pass}_pass.glsl"
            -o "${spv}"
        DEPENDS ${NR_FSR_SHADER_DEPENDENCIES}
        COMMENT "Compiling FSR 3.1 ${output}" VERBATIM)
    set(NR_FSR_SHADER_OUT ${NR_FSR_SHADER_OUT} "${spv}" PARENT_SCOPE)
endfunction()
foreach(nr_fsr_pass prepare_inputs luma_pyramid shading_change_pyramid shading_change
        prepare_reactivity luma_instability accumulate rcas debug_view autogen_reactive)
    nr_fsr_shader(${nr_fsr_pass} 0 "ffx_fsr3upscaler_${nr_fsr_pass}_pass")
endforeach()
nr_fsr_shader(accumulate 1 "ffx_fsr3upscaler_accumulate_pass_sharpen")
add_custom_target(NoorRayFsrShaders DEPENDS ${NR_FSR_SHADER_OUT})
add_dependencies(NoorRayFsr NoorRayFsrShaders)
target_compile_options(NoorRayFsr PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:--embed-dir=${NR_SHADER_BUILD_DIR}>")
file(GLOB NR_SHARED_SHADER_ABI CONFIGURE_DEPENDS
    "${NR_CORE_DIR}/Shared/*.h")
set(NR_SHADER_SRC "${NR_SHADER_DIR}/Viewport/Viewport.slang" "${NR_SHADER_DIR}/Viewport/ViewportBillboards.slang"
    "${NR_SHADER_DIR}/Raytracer/Raytracer.slang"
    "${NR_SHADER_DIR}/Raytracer/RaytracingAbi.slang"
    "${NR_SHADER_DIR}/Raytracer/RaytracingMiss.slang"
    "${NR_SHADER_DIR}/Raytracer/RaytracingHit.slang"
    "${NR_SHADER_DIR}/Raytracer/EmissionHit.slang"
    "${NR_SHADER_DIR}/Raytracer/OpacityAnyHit.slang"
    "${NR_SHADER_DIR}/Raytracer/GaussianAnyHit.slang"
    "${NR_SHADER_DIR}/Raytracer/GaussianHit.slang"
    "${NR_SHADER_DIR}/Raytracer/MaterialSVM.slang"
    "${NR_SHADER_DIR}/Raytracer/SceneSurface.slang"
    "${NR_SHADER_DIR}/Raytracer/Environment.slang"
    "${NR_SHADER_DIR}/Raytracer/Sampling.slang"
    ${NR_RAYTRACER_SHADER_INCLUDES}
    ${NR_REALTIME_SHADER_INCLUDES}
    "${NR_SHADER_DIR}/Gaussian/Gaussian.slang")
set(NR_SHADER_OUT "${NR_SHADER_BUILD_DIR}/Viewport/Viewport.spv" "${NR_SHADER_BUILD_DIR}/Viewport/ViewportBillboards.spv"
    "${NR_SHADER_BUILD_DIR}/Raytracer/Raytracer.spv"
    "${NR_SHADER_BUILD_DIR}/Raytracer/RaytracingMiss.spv"
    "${NR_SHADER_BUILD_DIR}/Raytracer/RaytracingHit.spv"
    "${NR_SHADER_BUILD_DIR}/Raytracer/EmissionHit.spv"
    "${NR_SHADER_BUILD_DIR}/Raytracer/OpacityAnyHit.spv"
    "${NR_SHADER_BUILD_DIR}/Raytracer/GaussianAnyHit.spv"
    "${NR_SHADER_BUILD_DIR}/Raytracer/GaussianHit.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeRaytracer.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeMiss.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeHit.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeSharcUpdate.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/SharcResolve.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeComposite.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeOutputAovs.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/Accumulate.spv")
set(NR_SHADER_VALIDATE_COMMANDS)
if(NR_SPIRV_VAL_EXECUTABLE)
    foreach(_spv IN LISTS NR_SHADER_OUT)
        list(APPEND NR_SHADER_VALIDATE_COMMANDS
            COMMAND "${NR_SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.4
                --scalar-block-layout "${_spv}")
    endforeach()
endif()
add_custom_command(OUTPUT ${NR_SHADER_OUT}
    COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${NR_SHADER_BUILD_DIR}/Viewport" "${NR_SHADER_BUILD_DIR}/Raytracer"
        "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Viewport/Viewport.slang"
        -entry main -target spirv -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/Viewport/Viewport.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Viewport/ViewportBillboards.slang"
        -entry vertMain -stage vertex -entry fragMain -stage fragment
        -entry lightIdClear -stage compute -entry lightIdStamp -stage compute
        -target spirv -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/Viewport/ViewportBillboards.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/Raytracer.slang"
        -entry main -stage raygeneration -target spirv
        -profile spirv_1_5+spvRayTracingKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/Raytracer/Raytracer.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/RaytracingMiss.slang"
        -entry main -stage miss -target spirv
        -profile spirv_1_5+spvRayTracingKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/Raytracer/RaytracingMiss.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/RaytracingHit.slang"
        -entry main -stage closesthit -target spirv
        -profile spirv_1_5+spvRayTracingKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/Raytracer/RaytracingHit.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/EmissionHit.slang"
        -entry main -stage closesthit -target spirv
        -profile spirv_1_5+spvRayTracingKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/Raytracer/EmissionHit.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/OpacityAnyHit.slang"
        -entry main -stage anyhit -target spirv
        -profile spirv_1_5+spvRayTracingKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/Raytracer/OpacityAnyHit.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/GaussianAnyHit.slang"
        -entry main -stage anyhit -target spirv
        -profile spirv_1_5+spvRayTracingKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/Raytracer/GaussianAnyHit.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/GaussianHit.slang"
        -entry main -stage closesthit -target spirv
        -profile spirv_1_5+spvRayTracingKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/Raytracer/GaussianHit.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeRaytracer.slang"
        ${NR_SHARC_SHADER_FLAGS} ${NR_NRD_SHADER_FLAGS} ${NR_RTXDI_SHADER_FLAGS} -DSHARC_QUERY=1
        -entry main -stage raygeneration -target spirv
        -profile spirv_1_5+spvRayTracingKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeRaytracer.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeMiss.slang"
        -entry main -stage miss -target spirv
        -profile spirv_1_5+spvRayTracingKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeMiss.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeHit.slang"
        ${NR_RTXDI_SHADER_FLAGS}
        -entry main -stage closesthit -target spirv
        -profile spirv_1_5+spvRayTracingKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeHit.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeRaytracer.slang"
        ${NR_SHARC_SHADER_FLAGS} ${NR_NRD_SHADER_FLAGS} ${NR_RTXDI_SHADER_FLAGS} -DSHARC_UPDATE=1
        -entry main -stage raygeneration -target spirv
        -profile spirv_1_5+spvRayTracingKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeSharcUpdate.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/SharcResolve.slang"
        ${NR_SHARC_SHADER_FLAGS} ${NR_RTXDI_SHADER_FLAGS}
        -entry main -stage compute -target spirv
        -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/SharcResolve.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeComposite.slang"
        ${NR_NRD_SHADER_FLAGS} ${NR_RTXDI_SHADER_FLAGS}
        -entry main -stage compute -target spirv
        -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeComposite.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeOutputAovs.slang"
        ${NR_RTXDI_SHADER_FLAGS}
        -entry main -stage compute -target spirv
        -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeOutputAovs.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/Accumulate.slang"
        ${NR_RTXDI_SHADER_FLAGS}
        -entry main -stage compute -target spirv
        -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/Accumulate.spv"
    ${NR_SHADER_VALIDATE_COMMANDS}
    DEPENDS ${NR_SHADER_SRC} ${NR_SHARED_SHADER_ABI} ${NR_SHARC_HEADERS}
        ${NR_RTXDI_HEADERS}
        "${PROJECT_SOURCE_DIR}/external/NRD/Shaders/NRD.hlsli"
    COMMENT "Compiling viewport shaders" VERBATIM)

# RTXDI passes: RtxdiPasses.slang built once per RTXDI_PASS_* entry point.
# Each is its own command so they compile in parallel.
set(NR_RTXDI_SHADER_OUT)
function(nr_rtxdi_pass pass stage output)
    set(spv "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/${output}.spv")
    set(profile spirv_1_5)
    if(stage STREQUAL "raygeneration")
        set(profile spirv_1_5+spvRayTracingKHR)
    endif()
    set(validate)
    if(NR_SPIRV_VAL_EXECUTABLE)
        set(validate COMMAND "${NR_SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.4
            --scalar-block-layout "${spv}")
    endif()
    add_custom_command(OUTPUT "${spv}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer"
        COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RtxdiPasses.slang"
            ${NR_RTXDI_SHADER_FLAGS} ${NR_NRD_SHADER_FLAGS} -DRTXDI_PASS_${pass}=1
            -entry main -stage ${stage} -target spirv -profile ${profile}
            -capability spvDescriptorHeapEXT -emit-spirv-directly
            -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
            -o "${spv}"
        ${validate}
        DEPENDS ${NR_SHADER_SRC} ${NR_SHARED_SHADER_ABI} ${NR_RTXDI_HEADERS}
            "${PROJECT_SOURCE_DIR}/external/NRD/Shaders/NRD.hlsli"
        COMMENT "Compiling RTXDI ${output}" VERBATIM)
    set(NR_RTXDI_SHADER_OUT ${NR_RTXDI_SHADER_OUT} "${spv}" PARENT_SCOPE)
endfunction()
nr_rtxdi_pass(PRESAMPLE_LIGHTS compute RtxdiPresampleLights)
nr_rtxdi_pass(PRESAMPLE_REGIR compute RtxdiPresampleReGIR)
nr_rtxdi_pass(DI_INITIAL raygeneration RtxdiDIInitial)
nr_rtxdi_pass(DI_TEMPORAL raygeneration RtxdiDITemporal)
nr_rtxdi_pass(DI_BOILING compute RtxdiDIBoiling)
nr_rtxdi_pass(DI_SPATIAL raygeneration RtxdiDISpatial)
nr_rtxdi_pass(DI_SHADE raygeneration RtxdiDIShade)
nr_rtxdi_pass(GI_TEMPORAL raygeneration RtxdiGITemporal)
nr_rtxdi_pass(GI_BOILING compute RtxdiGIBoiling)
nr_rtxdi_pass(GI_SPATIAL raygeneration RtxdiGISpatial)
nr_rtxdi_pass(GI_SHADE raygeneration RtxdiGIShade)
add_custom_target(NoorRayShaders DEPENDS ${NR_SHADER_OUT} ${NR_RTXDI_SHADER_OUT})

# ---------------------------------------------------------------------------
# NoorRay (standalone application)
# ---------------------------------------------------------------------------
if(NR_BUILD_APPLICATION)
    file(GLOB_RECURSE NR_APP_FILES CONFIGURE_DEPENDS "${NR_APP_DIR}/*.cpp" "${NR_APP_DIR}/*.h")
    add_executable(NoorRay ${NR_APP_FILES})
    set_target_properties(NoorRay PROPERTIES
        PREFIX "")
    add_dependencies(NoorRay NoorRayShaders)
    target_include_directories(NoorRay PRIVATE
        "${NR_APP_DIR}" "${NR_CORE_DIR}" "${PROJECT_SOURCE_DIR}/external/materialx/source")
    target_link_libraries(NoorRay PRIVATE
        libnoorray portable_file_dialogs glm imgui imnodeflow Vulkan::Vulkan
        SDL3::SDL3-static tinygltf tinyobjloader tinyexr stb gauss_forge_readers
        ImViewGuizmo GPUOpen::VulkanMemoryAllocator glaze TBB::tbb NoorRayMaterialX)
    if(NR_ENABLE_USD)
        target_link_libraries(NoorRay PRIVATE NoorRayUsdStandalone)
    endif()
    target_compile_definitions(NoorRay PRIVATE
        SDL_MAIN_HANDLED VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        NR_MATERIALX_STDLIB_DIR="${NR_MATERIALX_STDLIB_DIR}"
        # The editor resolves scene-relative asset paths against the source
        # tree's assets; the library itself carries no such path.
        NOORRAY_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(NoorRay PRIVATE -Wno-c23-extensions)
    endif()
endif()

# The Hydra delegate is an optional host plugin.  It is intentionally kept
# outside the default build because it needs Blender's ABI-matched OpenUSD
# SDK, but when enabled it links only against libnoorray and the NoorRHI API.
option(NR_BUILD_HYDRA "Build the hdnoorray Hydra render delegate" OFF)
if(NR_BUILD_HYDRA)
    add_subdirectory(src/hdnoorray)
endif()

# ---------------------------------------------------------------------------
# _pynoorray (Python bindings)
# ---------------------------------------------------------------------------
option(NR_BUILD_PYTHON "Build Python bindings" OFF)
if(NR_BUILD_PYTHON)
    # Do not pin the bindings to the build host's Python minor version.  The
    # extension ABI is selected by the interpreter/development package CMake
    # finds (or by a user supplied Python_EXECUTABLE).
    # Clear the stale cache value emitted by pre-Vulkan revisions, which
    # hard-coded a Python version that may no longer exist on the host.
    if(DEFINED Python_EXECUTABLE AND NOT EXISTS "${Python_EXECUTABLE}")
        unset(Python_EXECUTABLE CACHE)
    endif()
    find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
    add_subdirectory("${PROJECT_SOURCE_DIR}/external/nanobind"
        "${CMAKE_BINARY_DIR}/pynoorray/nanobind" EXCLUDE_FROM_ALL)
    file(GLOB NR_PYTHON_BINDINGS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/pynoorray/*Bindings.cpp")
    nanobind_add_module(_pynoorray
        src/pynoorray/PyNoorRay.cpp
        src/pynoorray/Bindings.h
        ${NR_PYTHON_BINDINGS}
        src/pynoorray/__init__.py)
    set_target_properties(_pynoorray PROPERTIES DEBUG_POSTFIX "")
    target_compile_features(_pynoorray PRIVATE cxx_std_23)
    target_include_directories(_pynoorray PRIVATE
        "${PROJECT_SOURCE_DIR}/src/pynoorray"
        "${NR_CORE_DIR}")
    target_link_libraries(_pynoorray PRIVATE
        "$<LINK_GROUP:RESCAN,libnoorray,NoorRayIO>")
    target_compile_definitions(_pynoorray PRIVATE
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
    add_custom_command(TARGET _pynoorray POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:_pynoorray>/pynoorray"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${PROJECT_SOURCE_DIR}/src/pynoorray/__init__.py"
            "$<TARGET_FILE_DIR:_pynoorray>/pynoorray/__init__.py"
        COMMAND ${CMAKE_COMMAND} -E copy
            "$<TARGET_FILE:_pynoorray>"
            "$<TARGET_FILE_DIR:_pynoorray>/pynoorray/"
        COMMENT "Copying pynoorray package")
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(NR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
