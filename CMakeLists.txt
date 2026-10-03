cmake_minimum_required(VERSION 3.25)

if(NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "Set CMAKE_CXX_COMPILER")
endif()
project(NoorRay VERSION 0.1.0 LANGUAGES C CXX)

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
# NoorRay::Core renders scenes a host builds in memory. The other libraries
# are optional: a host that does not link one can turn it off, and then none
# of its dependencies are configured or fetched.
option(NR_BUILD_IMPORT "Build NoorRay::Import (scene files, glTF, OBJ, PBRT, Gaussian splats)"
    ${PROJECT_IS_TOP_LEVEL})
option(NR_BUILD_SPECTRAL "Build NoorRay::Spectral (spectral reference path tracer)"
    ${PROJECT_IS_TOP_LEVEL})
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

# Shaders, MaterialX's libraries and editor assets are compiled into the
# libraries as CMakeRC resources, so no binary depends on this source tree.
include(external/CMakeRC/CMakeRC.cmake)

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
nr_header_only(stb          "${PROJECT_SOURCE_DIR}/external/stb")
if(NR_BUILD_EDITOR_DEPENDENCIES)
    nr_header_only(ImViewGuizmo "${PROJECT_SOURCE_DIR}/external/ImViewGuizmo")
endif()
if(NR_BUILD_IMPORT)
    nr_header_only(tinygltf     "${PROJECT_SOURCE_DIR}/external/tinygltf")
    nr_header_only(tinyobjloader "${PROJECT_SOURCE_DIR}/external/tinyobjloader")
    nr_header_only(glaze        "${PROJECT_SOURCE_DIR}/external/glaze/include")

    add_library(tinyply STATIC external/tinyply/source/tinyply.cpp)
    set_target_properties(tinyply PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(tinyply PUBLIC "${PROJECT_SOURCE_DIR}/external/tinyply/source")
endif()

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
if(NR_BUILD_IMPORT)
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
endif()

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
# NRD downloads ShaderMake and MathLib unless their targets exist. ShaderMake is
# built in this project rather than as its own external tool build, and compiles
# NRD's SPIR-V with the Vulkan SDK's dxc instead of downloading one; on Linux it
# would only search PATH for it. FXC is only for DXBC, which NRD does not embed.
find_program(SHADERMAKE_DXC_VK_PATH dxc HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin" REQUIRED)
set(SHADERMAKE_TOOL OFF CACHE BOOL "" FORCE)
set(SHADERMAKE_FIND_DXC OFF CACHE BOOL "" FORCE)
set(SHADERMAKE_FIND_FXC OFF CACHE BOOL "" FORCE)
add_subdirectory(external/ShaderMake EXCLUDE_FROM_ALL)
add_subdirectory(external/MathLib EXCLUDE_FROM_ALL)
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
set(NR_FFX_SDK_SOURCES
    "${NR_FFX_SDK_DIR}/src/backends/vk/ffx_vk.cpp"
    "${NR_FFX_SDK_DIR}/src/components/fsr3upscaler/ffx_fsr3upscaler.cpp"
    "${NR_FFX_SDK_DIR}/src/shared/ffx_assert.cpp"
    "${NR_FFX_SDK_DIR}/src/shared/ffx_message.cpp"
    "${NR_FFX_SDK_DIR}/src/shared/ffx_object_management.cpp"
    "${NR_FFX_SDK_DIR}/src/shared/ffx_breadcrumbs_list.cpp")
add_library(NoorRayFsr STATIC ${NR_FFX_SDK_SOURCES}
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
target_compile_definitions(NoorRayFsr PUBLIC FFX_FSR3UPSCALER $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:FFX_GCC>)
# The SDK is third-party code written for MSVC: keep its warnings out of the
# build. FfxPlatform.h, force-included into its C++ sources, supplies the
# standard headers it uses without including and, for other compilers, the
# MSVC functions. A source property rather than a target option: Visual
# Studio generators would apply a language-conditional option to the C file.
target_compile_options(NoorRayFsr PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>)
if(MSVC)
    set_property(SOURCE ${NR_FFX_SDK_SOURCES} APPEND PROPERTY COMPILE_OPTIONS
        "/FI${NR_CORE_DIR}/Realtime/FfxPlatform.h")
else()
    set_property(SOURCE ${NR_FFX_SDK_SOURCES} APPEND PROPERTY COMPILE_OPTIONS
        -include "${NR_CORE_DIR}/Realtime/FfxPlatform.h")
endif()
target_link_libraries(NoorRayFsr PUBLIC Vulkan::Vulkan)
# The Vulkan backend places its bookkeeping structures in one scratch arena at
# 4-byte offsets. Optimized GCC code writes those structures with aligned SSE
# moves, which fault there. The backend only schedules and records FSR's
# dispatches, so it is built without optimization; everything else is not.
set_property(SOURCE "${NR_FFX_SDK_DIR}/src/backends/vk/ffx_vk.cpp" APPEND PROPERTY COMPILE_OPTIONS
    $<IF:$<CXX_COMPILER_ID:MSVC>,/Od,-O0>)

# ---------------------------------------------------------------------------
# NoorRay::Core - scene, realtime renderer, MaterialX materials
# ---------------------------------------------------------------------------
file(GLOB_RECURSE NR_CORE_FILES CONFIGURE_DEPENDS
    "${NR_CORE_DIR}/*.cpp" "${NR_CORE_DIR}/*.h")
# Everything the optional libraries below own, and the MaterialX sources that
# NoorRayMaterialX builds.
set(NR_CORE_X
    "/IO/" "/Scene/Import/" "/Materials/Shading/" "/Raytracing/SpectralRaytracer\\."
    "/Materials/MaterialX/MaterialXDocument\\.cpp$" "/Materials/MaterialX/SlangMaterial[A-Za-z]*\\.cpp$"
    "/Realtime/FsrShaderBlobs\\.cpp$")
foreach(p IN LISTS NR_CORE_X)
    list(FILTER NR_CORE_FILES EXCLUDE REGEX "${p}")
endforeach()
add_library(libnoorray STATIC ${NR_CORE_FILES})
add_library(NoorRay::Core ALIAS libnoorray)
set_target_properties(libnoorray PROPERTIES
    OUTPUT_NAME noorray
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(libnoorray PUBLIC cxx_std_23)
target_include_directories(libnoorray PUBLIC "${NR_CORE_DIR}")
target_compile_definitions(libnoorray PUBLIC
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
# Public: what libnoorray's own headers include. Everything else is an
# implementation detail; as a static library it still reaches the link line.
target_link_libraries(libnoorray
    PUBLIC
        NoorRHI::NoorRHI glm Vulkan::Vulkan MaterialXCore Rtxdi
    PRIVATE
        TBB::tbb NoorRayMaterialX MaterialXFormat NRD NoorRayFsr NoorRayShaderResources)

# ---------------------------------------------------------------------------
# NoorRay::IO - image files (stb formats, OpenEXR)
# ---------------------------------------------------------------------------
add_library(NoorRayIO STATIC
    src/libnoorray/IO/BitmapReader.cpp
    src/libnoorray/IO/BitmapWriter.cpp
    src/libnoorray/IO/StbImpl.cpp
    src/libnoorray/IO/TextureReader.cpp)
add_library(NoorRay::IO ALIAS NoorRayIO)
set_target_properties(NoorRayIO PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(NoorRayIO PUBLIC libnoorray PRIVATE stb tinyexr)

# ---------------------------------------------------------------------------
# NoorRay::Import - scene files and the formats they reference
# ---------------------------------------------------------------------------
if(NR_BUILD_IMPORT)
    file(GLOB NR_IMPORT_FILES CONFIGURE_DEPENDS "${NR_CORE_DIR}/Scene/Import/*.cpp")
    # The stub implements the same entry points as the USD reader and writer,
    # so their callers need no conditional compilation.
    if(NR_ENABLE_USD)
        list(FILTER NR_IMPORT_FILES EXCLUDE REGEX "/SceneUsdDisabled\\.cpp$")
    else()
        list(FILTER NR_IMPORT_FILES EXCLUDE REGEX "/SceneUsd\\.cpp$")
    endif()
    add_library(NoorRayImport STATIC ${NR_IMPORT_FILES})
    add_library(NoorRay::Import ALIAS NoorRayImport)
    set_target_properties(NoorRayImport PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(NoorRayImport
        PUBLIC libnoorray NoorRayIO
        PRIVATE tinygltf tinyobjloader tinyply gauss_forge_readers glaze TBB::tbb MaterialXFormat)
    if(NR_ENABLE_USD)
        target_link_libraries(NoorRayImport PUBLIC NoorRayUsd)
    endif()
elseif(NR_ENABLE_USD)
    message(FATAL_ERROR "NR_ENABLE_USD needs NR_BUILD_IMPORT")
endif()

# ---------------------------------------------------------------------------
# NoorRay::Spectral - spectral reference path tracer
# ---------------------------------------------------------------------------
# Its own library because its RGB-to-spectrum tables alone are ~10 MB.
if(NR_BUILD_SPECTRAL)
    file(GLOB NR_SPECTRAL_FILES CONFIGURE_DEPENDS "${NR_CORE_DIR}/Materials/Shading/*.cpp")
    add_library(NoorRaySpectral STATIC
        "${NR_CORE_DIR}/Raytracing/SpectralRaytracer.cpp" ${NR_SPECTRAL_FILES})
    add_library(NoorRay::Spectral ALIAS NoorRaySpectral)
    set_target_properties(NoorRaySpectral PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(NoorRaySpectral PUBLIC libnoorray PRIVATE NoorRayShaderResources)
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

# The MaterialX libraries are embedded so a NoorRay binary does not depend on
# this source tree; they are unpacked to the user cache on first use because
# MaterialX reads definitions and generator sources only from files. Slang
# generation needs genslang plus the genglsl sources it includes.
file(GLOB_RECURSE NR_MATERIALX_LIBRARY_FILES CONFIGURE_DEPENDS
    "${NR_MATERIALX_STDLIB_DIR}/*.mtlx"
    "${NR_MATERIALX_STDLIB_DIR}/*.glsl"
    "${NR_MATERIALX_STDLIB_DIR}/*.slang")
list(FILTER NR_MATERIALX_LIBRARY_FILES EXCLUDE REGEX "/gen(mdl|msl|osl)/")
cmrc_add_resource_library(NoorRayMaterialXLibrary NAMESPACE noorray_materialx
    WHENCE "${NR_MATERIALX_STDLIB_DIR}" ${NR_MATERIALX_LIBRARY_FILES})
set_target_properties(NoorRayMaterialXLibrary PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(NoorRayMaterialX STATIC
    src/libnoorray/Materials/MaterialX/MaterialXDocument.cpp
    src/libnoorray/Materials/MaterialX/SlangMaterialGenerator.cpp
    src/libnoorray/Materials/MaterialX/SlangMaterialCompiler.cpp)
target_compile_features(NoorRayMaterialX PUBLIC cxx_std_23)
target_include_directories(NoorRayMaterialX PUBLIC "${NR_CORE_DIR}")
# Generated materials are compiled at runtime with the Slang compiler
# library, built statically from external/slang at the revision of the SDK
# slangc that builds the prebuilt shaders. The SDK ships it only as a shared
# object, which would tie every binary to a Slang .so at runtime.
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/slang/external/lua/lua.h")
    message(FATAL_ERROR "external/slang is empty. Run: git submodule update --init --recursive external/slang")
endif()
set(SLANG_LIB_TYPE STATIC CACHE STRING "" FORCE)
set(SLANG_EMBED_CORE_MODULE ON CACHE BOOL "" FORCE)
set(SLANG_EMBED_CORE_MODULE_SOURCE OFF CACHE BOOL "" FORCE)
set(SLANG_ENABLE_RELEASE_DEBUG_INFO OFF CACHE BOOL "" FORCE)
foreach(_slang_component GFX SLANGD SLANGC SLANGI SLANGRT SLANG_GLSLANG SLANG_PROXY SLANG_RHI
        TESTS EXAMPLES REPLAYER DXIL CUDA OPTIX XLIB AFTERMATH)
    set(SLANG_ENABLE_${_slang_component} OFF CACHE BOOL "" FORCE)
endforeach()
set(SLANG_SLANG_LLVM_FLAVOR DISABLE CACHE STRING "" FORCE)
set(SLANG_EXCLUDE_DAWN ON CACHE BOOL "" FORCE)
set(SLANG_EXCLUDE_TINT ON CACHE BOOL "" FORCE)
add_subdirectory(external/slang EXCLUDE_FROM_ALL)
# Slang puts these back into ALL although its directory is excluded; neither the
# slang command line nor its C++ parser is needed.
set_target_properties(slang-dispatcher slang-cpp-parser PROPERTIES EXCLUDE_FROM_ALL ON)
target_link_libraries(NoorRayMaterialX PUBLIC
    glm MaterialXGenSlang MaterialXGenShader MaterialXFormat MaterialXCore)
target_link_libraries(NoorRayMaterialX PRIVATE slang NoorRayMaterialXLibrary NoorRayShaderResources)
# Generated materials are validated before they reach the driver, whose SPIR-V
# compiler can crash on invalid modules. SPIRV-Tools is the copy Slang vendors;
# its headers come from the SPIRV-Headers Slang already added.
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
add_subdirectory(external/slang/external/spirv-tools EXCLUDE_FROM_ALL)
target_link_libraries(NoorRayMaterialX PRIVATE SPIRV-Tools-static)
set_target_properties(NoorRayMaterialX PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Vulkan shaders (Viewport, ViewportBillboards)
# ---------------------------------------------------------------------------
set(NR_SHADER_DIR "${NR_CORE_DIR}/Shaders")
# Compiled SPIR-V is a build product: it lives in the binary tree and reaches
# the libraries as NoorRayShaderResources, by its path relative to this directory.
set(NR_SHADER_BUILD_DIR "${PROJECT_BINARY_DIR}/shaders")
file(GLOB NR_RAYTRACER_SHADER_INCLUDES CONFIGURE_DEPENDS
    "${NR_SHADER_DIR}/Raytracer/*.slang")
file(GLOB NR_REALTIME_SHADER_INCLUDES CONFIGURE_DEPENDS
    "${NR_SHADER_DIR}/RealtimeRaytracer/*.slang")
# NRD front-end helpers (NRD.hlsli) packing the denoiser inputs. Its
# NRDConfig.hlsli is generated when external/NRD is configured below.
set(NR_NRD_SHADER_FLAGS -I "${PROJECT_SOURCE_DIR}/external/NRD/Shaders")
# RTXDI's resampling headers, included by the realtime shaders through
# RtxdiBridge.slang.
set(NR_RTXDI_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/RTXDI-Library/Include")
file(GLOB_RECURSE NR_RTXDI_HEADERS CONFIGURE_DEPENDS
    "${NR_RTXDI_INCLUDE_DIR}/Rtxdi/*.h" "${NR_RTXDI_INCLUDE_DIR}/Rtxdi/*.hlsli")
set(NR_RTXDI_SHADER_FLAGS -I "${NR_RTXDI_INCLUDE_DIR}")
# Everything that includes RTXDI's path tracer (Rtxdi/PT) needs these.
set(NR_RTXDI_PT_SHADER_FLAGS -DRTXDI_SLANG=1 -DRTXDI_PATHTRACER_USE_REORDERING=0)

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
target_link_libraries(NoorRayFsr PRIVATE NoorRayShaderResources)
file(GLOB NR_SHARED_SHADER_ABI CONFIGURE_DEPENDS
    "${NR_CORE_DIR}/Shared/*.h")
set(NR_SHADER_SRC "${NR_SHADER_DIR}/Viewport/Viewport.slang" "${NR_SHADER_DIR}/Viewport/ViewportBillboards.slang"
    "${NR_SHADER_DIR}/Viewport/ViewportVolumes.slang"
    "${NR_SHADER_DIR}/Raytracer/Raytracer.slang"
    "${NR_SHADER_DIR}/Raytracer/RaytracingAbi.slang"
    "${NR_SHADER_DIR}/Raytracer/RaytracingMiss.slang"
    "${NR_SHADER_DIR}/Raytracer/RaytracingHit.slang"
    "${NR_SHADER_DIR}/Raytracer/EmissionHit.slang"
    "${NR_SHADER_DIR}/Raytracer/OpacityAnyHit.slang"
    "${NR_SHADER_DIR}/Raytracer/GaussianAnyHit.slang"
    "${NR_SHADER_DIR}/Raytracer/GaussianHit.slang"
    "${NR_SHADER_DIR}/Raytracer/GaussianSplat.slang"
    "${NR_SHADER_DIR}/Raytracer/SceneSurface.slang"
    "${NR_SHADER_DIR}/Raytracer/Environment.slang"
    "${NR_SHADER_DIR}/Raytracer/Sampling.slang"
    ${NR_RAYTRACER_SHADER_INCLUDES}
    ${NR_REALTIME_SHADER_INCLUDES}
    "${NR_SHADER_DIR}/Gaussian/Gaussian.slang"
    "${NR_SHADER_DIR}/Mesh/SplineMesh.slang"
    "${NR_SHADER_DIR}/Mesh/LandscapeMesh.slang")
set(NR_SHADER_OUT "${NR_SHADER_BUILD_DIR}/Viewport/Viewport.spv" "${NR_SHADER_BUILD_DIR}/Viewport/ViewportBillboards.spv"
    "${NR_SHADER_BUILD_DIR}/Viewport/ViewportVolumes.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeRaytracer.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeRaytracerLayered.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeGBuffer.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeGBufferLayered.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeMiss.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeShadowMiss.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/DefaultMaterialHit.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeComposite.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeCompositeLayered.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeOutputAovs.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimePick.spv"
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/Accumulate.spv"
    "${NR_SHADER_BUILD_DIR}/Mesh/SplineMesh.spv"
    "${NR_SHADER_BUILD_DIR}/Mesh/LandscapeMesh.spv")
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
        "${NR_SHADER_BUILD_DIR}/Viewport" "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer"
        "${NR_SHADER_BUILD_DIR}/Mesh"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Viewport/Viewport.slang"
        -entry main -target spirv -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/Viewport/Viewport.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Viewport/ViewportBillboards.slang"
        -entry vertMain -stage vertex -entry fragMain -stage fragment
        -entry lightIdClear -stage compute -entry lightIdStamp -stage compute
        -target spirv -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/Viewport/ViewportBillboards.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Viewport/ViewportVolumes.slang"
        -entry vertMain -stage vertex -entry fragMain -stage fragment
        -target spirv -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/Viewport/ViewportVolumes.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeRaytracer.slang"
        ${NR_NRD_SHADER_FLAGS} ${NR_RTXDI_SHADER_FLAGS}
        -entry main -stage raygeneration -target spirv
        -profile spirv_1_5+spvRayTracingKHR+spvRayQueryKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeRaytracer.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeRaytracer.slang"
        ${NR_NRD_SHADER_FLAGS} ${NR_RTXDI_SHADER_FLAGS} -DREALTIME_LAYERS=1
        -entry main -stage raygeneration -target spirv
        -profile spirv_1_5+spvRayTracingKHR+spvRayQueryKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeRaytracerLayered.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeRaytracer.slang"
        ${NR_NRD_SHADER_FLAGS} ${NR_RTXDI_SHADER_FLAGS} -DREALTIME_GBUFFER=1
        -entry main -stage raygeneration -target spirv
        -profile spirv_1_5+spvRayTracingKHR+spvRayQueryKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeGBuffer.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeRaytracer.slang"
        ${NR_NRD_SHADER_FLAGS} ${NR_RTXDI_SHADER_FLAGS} -DREALTIME_GBUFFER=1 -DREALTIME_LAYERS=1
        -entry main -stage raygeneration -target spirv
        -profile spirv_1_5+spvRayTracingKHR+spvRayQueryKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeGBufferLayered.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeMiss.slang"
        -entry main -stage miss -target spirv
        -profile spirv_1_5+spvRayTracingKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeMiss.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeShadowMiss.slang"
        -entry main -stage miss -target spirv
        -profile spirv_1_5+spvRayTracingKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeShadowMiss.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/DefaultMaterialHit.slang"
        -entry main -stage closesthit -entry anyHit -stage anyhit
        -entry shadowAnyHit -stage anyhit -target spirv
        -profile spirv_1_5+spvRayTracingKHR
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/DefaultMaterialHit.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeComposite.slang"
        ${NR_NRD_SHADER_FLAGS} ${NR_RTXDI_SHADER_FLAGS}
        -entry main -stage compute -target spirv
        -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeComposite.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeComposite.slang"
        ${NR_NRD_SHADER_FLAGS} ${NR_RTXDI_SHADER_FLAGS} -DREALTIME_LAYERED=1
        -entry main -stage compute -target spirv
        -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeCompositeLayered.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeOutputAovs.slang"
        ${NR_RTXDI_SHADER_FLAGS}
        -entry main -stage compute -target spirv
        -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimeOutputAovs.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RealtimeOutputAovs.slang"
        ${NR_RTXDI_SHADER_FLAGS} -DREALTIME_PICK=1
        -entry main -stage compute -target spirv
        -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/RealtimePick.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/Accumulate.slang"
        ${NR_RTXDI_SHADER_FLAGS}
        -entry main -stage compute -target spirv
        -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/Accumulate.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Mesh/SplineMesh.slang"
        -entry main -stage compute -target spirv
        -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/Mesh/SplineMesh.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Mesh/LandscapeMesh.slang"
        -entry main -stage compute -target spirv
        -profile spirv_1_5
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
        -o "${NR_SHADER_BUILD_DIR}/Mesh/LandscapeMesh.spv"
    ${NR_SHADER_VALIDATE_COMMANDS}
    DEPENDS ${NR_SHADER_SRC} ${NR_SHARED_SHADER_ABI}
        ${NR_RTXDI_HEADERS}
        "${PROJECT_SOURCE_DIR}/external/NRD/Shaders/NRD.hlsli"
    COMMENT "Compiling viewport shaders" VERBATIM)

# RTXDI passes: RtxdiPasses.slang built once per RTXDI_PASS_* entry point.
# Each is its own command so they compile in parallel.
set(NR_RTXDI_SHADER_OUT)
# The passes that replay paths or trace final visibility are ray-generation
# stages of the realtime pipeline; the others are compute shaders, which test
# RTXDI's conservative visibility with inline ray queries.
function(nr_rtxdi_pass pass output stage)
    set(pt_flags)
    set(pt_depends)
    if(pass MATCHES "^PT_" OR pass STREQUAL "INITIAL")
        set(pt_flags ${NR_RTXDI_PT_SHADER_FLAGS})
        if(NOT pass STREQUAL "PT_BOILING")
            set(pt_depends "${NR_SHADER_DIR}/RealtimeRaytracer/RtxdiPTPathTraceBody.inc")
        endif()
    endif()
    if(stage STREQUAL "raygeneration")
        set(profile spirv_1_5+spvRayTracingKHR+spvRayQueryKHR)
    else()
        set(profile spirv_1_5+spvRayQueryKHR)
    endif()
    if(pass MATCHES "^(DI|PT)_BOILING$")
        string(APPEND profile "+spvGroupNonUniform+spvGroupNonUniformBallot+spvGroupNonUniformArithmetic")
    endif()
    set(spv "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/${output}.spv")
    set(validate)
    if(NR_SPIRV_VAL_EXECUTABLE)
        set(validate COMMAND "${NR_SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.4
            --scalar-block-layout "${spv}")
    endif()
    add_custom_command(OUTPUT "${spv}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer"
        COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/RealtimeRaytracer/RtxdiPasses.slang"
            ${pt_flags} ${NR_RTXDI_SHADER_FLAGS} ${NR_NRD_SHADER_FLAGS}
            -DRTXDI_PASS_${pass}=1
            -entry main -stage ${stage} -target spirv -profile ${profile}
            -capability spvDescriptorHeapEXT -emit-spirv-directly
            -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
            -o "${spv}"
        ${validate}
        DEPENDS ${NR_SHADER_SRC} ${NR_SHARED_SHADER_ABI} ${NR_RTXDI_HEADERS} ${pt_depends}
            "${PROJECT_SOURCE_DIR}/external/NRD/Shaders/NRD.hlsli"
        COMMENT "Compiling RTXDI ${output}" VERBATIM)
    set(NR_RTXDI_SHADER_OUT ${NR_RTXDI_SHADER_OUT} "${spv}" PARENT_SCOPE)
endfunction()
nr_rtxdi_pass(PRESAMPLE_LIGHTS RtxdiPresampleLights compute)
nr_rtxdi_pass(PRESAMPLE_REGIR RtxdiPresampleReGIR compute)
nr_rtxdi_pass(PRESAMPLE_ENVIRONMENT RtxdiPresampleEnvironment compute)
nr_rtxdi_pass(INITIAL RtxdiInitial raygeneration)
nr_rtxdi_pass(DI_TEMPORAL RtxdiDITemporal compute)
nr_rtxdi_pass(DI_BOILING RtxdiDIBoiling compute)
nr_rtxdi_pass(DI_SPATIAL RtxdiDISpatial compute)
nr_rtxdi_pass(PT_TEMPORAL RtxdiPTTemporal raygeneration)
nr_rtxdi_pass(PT_BOILING RtxdiPTBoiling compute)
nr_rtxdi_pass(PT_SPATIAL RtxdiPTSpatial raygeneration)
nr_rtxdi_pass(SHADE RtxdiShade raygeneration)
add_custom_target(NoorRayShaders DEPENDS ${NR_SHADER_OUT} ${NR_RTXDI_SHADER_OUT})

# Materials compile their hit stages at runtime from MaterialHit.slang. The
# material compiler has no file system, so the file and everything it
# includes are preprocessed into one source it embeds.
set(NR_MATERIAL_HIT_SOURCE
    "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer/MaterialHit.preprocessed.slang")
add_custom_command(OUTPUT "${NR_MATERIAL_HIT_SOURCE}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${NR_SHADER_BUILD_DIR}/RealtimeRaytracer"
    COMMAND "${CMAKE_COMMAND}" -DSLANGC=${NR_SLANGC_EXECUTABLE}
        -DINPUT=${NR_SHADER_DIR}/RealtimeRaytracer/MaterialHit.slang
        -DOUTPUT=${NR_MATERIAL_HIT_SOURCE}
        -P "${PROJECT_SOURCE_DIR}/cmake/PreprocessSlang.cmake"
    DEPENDS ${NR_SHADER_SRC} ${NR_SHARED_SHADER_ABI} "${PROJECT_SOURCE_DIR}/cmake/PreprocessSlang.cmake"
    COMMENT "Preprocessing MaterialHit.slang" VERBATIM)

# Spectral path tracer shaders, compiled only for NoorRay::Spectral.
if(NR_BUILD_SPECTRAL)
    set(NR_SPECTRAL_SHADER_OUT)
    foreach(_shader Raytracer RaytracingMiss RaytracingHit EmissionHit OpacityAnyHit
            GaussianAnyHit GaussianHit)
        list(APPEND NR_SPECTRAL_SHADER_OUT "${NR_SHADER_BUILD_DIR}/Raytracer/${_shader}.spv")
    endforeach()
    set(NR_SPECTRAL_SHADER_VALIDATE_COMMANDS)
    if(NR_SPIRV_VAL_EXECUTABLE)
        foreach(_spv IN LISTS NR_SPECTRAL_SHADER_OUT)
            list(APPEND NR_SPECTRAL_SHADER_VALIDATE_COMMANDS
                COMMAND "${NR_SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.4
                    --scalar-block-layout "${_spv}")
        endforeach()
    endif()
    add_custom_command(OUTPUT ${NR_SPECTRAL_SHADER_OUT}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NR_SHADER_BUILD_DIR}/Raytracer"
        COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/Raytracer.slang"
            -entry main -stage raygeneration -target spirv
            -profile spirv_1_5+spvRayTracingKHR
            -capability spvDescriptorHeapEXT -emit-spirv-directly
            -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
            -o "${NR_SHADER_BUILD_DIR}/Raytracer/Raytracer.spv"
        COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/RaytracingMiss.slang"
            -entry main -stage miss -target spirv
            -profile spirv_1_5+spvRayTracingKHR
            -capability spvDescriptorHeapEXT -emit-spirv-directly
            -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
            -o "${NR_SHADER_BUILD_DIR}/Raytracer/RaytracingMiss.spv"
        COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/RaytracingHit.slang"
            -entry main -stage closesthit -target spirv
            -profile spirv_1_5+spvRayTracingKHR
            -capability spvDescriptorHeapEXT -emit-spirv-directly
            -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
            -o "${NR_SHADER_BUILD_DIR}/Raytracer/RaytracingHit.spv"
        COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/EmissionHit.slang"
            -entry main -stage closesthit -target spirv
            -profile spirv_1_5+spvRayTracingKHR
            -capability spvDescriptorHeapEXT -emit-spirv-directly
            -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
            -o "${NR_SHADER_BUILD_DIR}/Raytracer/EmissionHit.spv"
        COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/OpacityAnyHit.slang"
            -entry main -stage anyhit -target spirv
            -profile spirv_1_5+spvRayTracingKHR
            -capability spvDescriptorHeapEXT -emit-spirv-directly
            -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
            -o "${NR_SHADER_BUILD_DIR}/Raytracer/OpacityAnyHit.spv"
        COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/GaussianAnyHit.slang"
            -entry main -stage anyhit -target spirv
            -profile spirv_1_5+spvRayTracingKHR
            -capability spvDescriptorHeapEXT -emit-spirv-directly
            -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
            -o "${NR_SHADER_BUILD_DIR}/Raytracer/GaussianAnyHit.spv"
        COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/GaussianHit.slang"
            -entry main -stage closesthit -target spirv
            -profile spirv_1_5+spvRayTracingKHR
            -capability spvDescriptorHeapEXT -emit-spirv-directly
            -matrix-layout-row-major -fvk-use-entrypoint-name -fvk-use-c-layout -Wno-30081
            -o "${NR_SHADER_BUILD_DIR}/Raytracer/GaussianHit.spv"
        ${NR_SPECTRAL_SHADER_VALIDATE_COMMANDS}
        DEPENDS ${NR_SHADER_SRC} ${NR_SHARED_SHADER_ABI}
        COMMENT "Compiling spectral path tracer shaders" VERBATIM)
    add_custom_target(NoorRaySpectralShaders DEPENDS ${NR_SPECTRAL_SHADER_OUT})
endif()

# Every compiled shader, the Slang sources the material compiler builds on, and
# the billboard textures, as the CMakeRC namespace noorray_shaders
# (Realtime/ShaderLoading.h).
cmrc_add_resource_library(NoorRayShaderResources NAMESPACE noorray_shaders
    WHENCE "${NR_SHADER_BUILD_DIR}"
    ${NR_SHADER_OUT} ${NR_RTXDI_SHADER_OUT} ${NR_FSR_SHADER_OUT} ${NR_SPECTRAL_SHADER_OUT}
    "${NR_MATERIAL_HIT_SOURCE}")
cmrc_add_resources(NoorRayShaderResources WHENCE "${NR_SHADER_DIR}"
    "${NR_SHADER_DIR}/RealtimeRaytracer/MaterialInterface.slang")
# The viewport's billboard textures, generated by tools/billboards.
file(GLOB NR_BILLBOARD_TEXTURES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/assets/billboards/*.msdf")
cmrc_add_resources(NoorRayShaderResources WHENCE "${PROJECT_SOURCE_DIR}/assets"
    ${NR_BILLBOARD_TEXTURES})
set_target_properties(NoorRayShaderResources PROPERTIES POSITION_INDEPENDENT_CODE ON)
# The shader commands belong to these targets; depending on them keeps a
# Makefile build from running a command twice.
add_dependencies(NoorRayShaderResources NoorRayShaders NoorRayFsrShaders)
if(NR_BUILD_SPECTRAL)
    add_dependencies(NoorRayShaderResources NoorRaySpectralShaders)
endif()

# ---------------------------------------------------------------------------
# NoorRay (standalone application)
# ---------------------------------------------------------------------------
if(NR_BUILD_APPLICATION)
    if(NOT NR_BUILD_IMPORT)
        message(FATAL_ERROR "NR_BUILD_APPLICATION needs NR_BUILD_IMPORT")
    endif()
    file(GLOB_RECURSE NR_APP_FILES CONFIGURE_DEPENDS "${NR_APP_DIR}/*.cpp" "${NR_APP_DIR}/*.h")
    add_executable(NoorRay ${NR_APP_FILES})
    set_target_properties(NoorRay PROPERTIES
        PREFIX "")
    target_include_directories(NoorRay PRIVATE
        "${NR_APP_DIR}" "${NR_CORE_DIR}" "${PROJECT_SOURCE_DIR}/external/materialx/source")
    target_link_libraries(NoorRay PRIVATE
        libnoorray NoorRayIO NoorRayImport portable_file_dialogs glm imgui imnodeflow
        Vulkan::Vulkan SDL3::SDL3-static ImViewGuizmo TBB::tbb NoorRayMaterialX)
    if(NR_ENABLE_USD)
        target_link_libraries(NoorRay PRIVATE NoorRayUsdStandalone)
    endif()
    target_compile_definitions(NoorRay PRIVATE
        SDL_MAIN_HANDLED VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        # The editor resolves scene-relative asset paths against the source
        # tree's assets; the library itself carries no such path.
        NOORRAY_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
    cmrc_add_resource_library(NoorRayAppAssets NAMESPACE noorray_app
        WHENCE "${PROJECT_SOURCE_DIR}/assets"
        "${PROJECT_SOURCE_DIR}/assets/fonts/Inter.ttf" "${PROJECT_SOURCE_DIR}/assets/imgui.ini")
    target_link_libraries(NoorRay PRIVATE NoorRayAppAssets)
endif()

# The Hydra delegate is an optional host plugin.  It is intentionally kept
# outside the default build because it needs Blender's ABI-matched OpenUSD
# SDK, but when enabled it links only against libnoorray and the NoorRHI API.
option(NR_BUILD_HYDRA "Build the hdnoorray Hydra render delegate" OFF)
if(NR_BUILD_HYDRA)
    if(NOT NR_BUILD_IMPORT)
        message(FATAL_ERROR "NR_BUILD_HYDRA needs NR_BUILD_IMPORT")
    endif()
    add_subdirectory(src/hdnoorray)
endif()

# ---------------------------------------------------------------------------
# _pynoorray (Python bindings)
# ---------------------------------------------------------------------------
option(NR_BUILD_PYTHON "Build Python bindings" OFF)
if(NR_BUILD_PYTHON)
    if(NOT NR_BUILD_IMPORT)
        message(FATAL_ERROR "NR_BUILD_PYTHON needs NR_BUILD_IMPORT")
    endif()
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
    target_link_libraries(_pynoorray PRIVATE libnoorray NoorRayIO NoorRayImport)
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
    if(NOT NR_BUILD_IMPORT OR NOT NR_BUILD_SPECTRAL)
        message(FATAL_ERROR "NR_BUILD_TESTS needs NR_BUILD_IMPORT and NR_BUILD_SPECTRAL")
    endif()
    enable_testing()
    add_subdirectory(tests)
endif()
