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
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_SKIP_BUILD_RPATH ON)

set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)

set(NR_CORE_DIR "${PROJECT_SOURCE_DIR}/src/libnoorray")
set(NR_APP_DIR "${PROJECT_SOURCE_DIR}/src/app")

# USD provides scene import/export only. Hosts that just drive the renderer -
# an embedded viewport, for instance - never reach that path, and USD is by far
# the heaviest thing in the dependency graph: it pulls a Blender-provided
# libusd_ms.so and libpython into every link. Keeping it optional lets those
# hosts build and ship without either.
option(NR_ENABLE_USD "Build USD scene import/export" ON)
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
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(external/SDL)
if(TARGET SDL3-static)
    set_target_properties(SDL3-static PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
add_subdirectory(external/glm)
add_subdirectory(external/portable-file-dialogs)
add_subdirectory(external/VulkanMemoryAllocator)

# ---------------------------------------------------------------------------
# Header-only libraries
# ---------------------------------------------------------------------------
nr_header_only(tinygltf     "${PROJECT_SOURCE_DIR}/external/tinygltf")
nr_header_only(tinyobjloader "${PROJECT_SOURCE_DIR}/external/tinyobjloader")
nr_header_only(stb          "${PROJECT_SOURCE_DIR}/external/stb")
nr_header_only(ImViewGuizmo "${PROJECT_SOURCE_DIR}/external/ImViewGuizmo")
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
find_program(NR_SLANGC_EXECUTABLE NAMES slangc
    HINTS "$ENV{VULKAN_SDK}/bin" "/home/marcel/Programs/VulkanSDK/current/bin"
    NO_CACHE REQUIRED)
set(Vulkan_INCLUDE_DIR "${Vulkan_INCLUDE_DIR}" CACHE PATH "")

# ---------------------------------------------------------------------------
# ImGui + ImGuizmo
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# NoorRayIO (image I/O, stb wrapper)
# ---------------------------------------------------------------------------
add_library(NoorRayIO STATIC
    src/libnoorray/IO/BitmapReader.cpp
    src/libnoorray/IO/BitmapWriter.cpp
    src/libnoorray/StbImpl.cpp)
set_target_properties(NoorRayIO PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(NoorRayIO PUBLIC cxx_std_23)
target_include_directories(NoorRayIO PUBLIC "${NR_CORE_DIR}")
target_link_libraries(NoorRayIO PUBLIC glm stb tinyexr)

# ---------------------------------------------------------------------------
# gpu - small Vulkan graphics abstraction
# ---------------------------------------------------------------------------
add_library(NoorRayVma STATIC src/gpu/src/vma_impl.cpp)
set_target_properties(NoorRayVma PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(NoorRayVma PUBLIC cxx_std_23)
target_link_libraries(NoorRayVma PUBLIC Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator)

add_subdirectory(src/gpu)

# ---------------------------------------------------------------------------
# libnoorray (core engine library)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE NR_CORE_FILES CONFIGURE_DEPENDS
    "${NR_CORE_DIR}/*.cpp" "${NR_CORE_DIR}/*.h")
set(NR_CORE_X
    "/Materials/MaterialX/MaterialXDocument\\.cpp$"
    "/Materials/Shading/EnergyLut/GenerateEnergyLuts\\.cpp$" "/Materials/Shading/EnergyLut\\.cpp$"
    "/IO/BitmapReader\\.cpp$" "/IO/BitmapWriter\\.cpp$" "/StbImpl\\.cpp$"
    "/Materials/SVM/references/")
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
add_dependencies(libnoorray NoorRayShaders)
set_target_properties(libnoorray PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(libnoorray PRIVATE -Wno-c23-extensions)
endif()
target_compile_features(libnoorray PUBLIC cxx_std_23)
target_include_directories(libnoorray PUBLIC "${NR_CORE_DIR}")
target_compile_definitions(libnoorray PUBLIC
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    NOORRAY_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")
target_link_libraries(libnoorray PUBLIC
    NoorRayIO noorray_gpu NoorRayVma portable_file_dialogs glm Vulkan::Vulkan
    tinygltf tinyobjloader tinyply tinyexr stb gauss_forge_readers
    GPUOpen::VulkanMemoryAllocator glaze TBB::tbb
    NoorRayMaterialX MaterialXFormat MaterialXCore)
if(NR_ENABLE_USD)
    target_link_libraries(libnoorray PUBLIC NoorRayUsd)
endif()

# ---------------------------------------------------------------------------
# MaterialX
# ---------------------------------------------------------------------------
set(NR_MATERIALX_PINNED_VERSION "1.39.4")
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
add_subdirectory(external/materialx)
foreach(t MaterialXCore MaterialXFormat MaterialXGenShader)
    if(TARGET ${t})
        set_target_properties(${t} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
endforeach()
set(NR_MATERIALX_STDLIB_DIR "${PROJECT_SOURCE_DIR}/external/materialx/libraries")

add_library(NoorRayMaterialX STATIC
    src/libnoorray/Materials/MaterialX/MaterialXDocument.cpp)
target_compile_features(NoorRayMaterialX PUBLIC cxx_std_23)
target_include_directories(NoorRayMaterialX PUBLIC "${NR_CORE_DIR}")
target_link_libraries(NoorRayMaterialX PUBLIC
    glm MaterialXGenShader MaterialXFormat MaterialXCore)
target_compile_definitions(NoorRayMaterialX PRIVATE
    NR_MATERIALX_STDLIB_DIR="${NR_MATERIALX_STDLIB_DIR}")
set_target_properties(NoorRayMaterialX PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Vulkan shaders (Viewport, ViewportBillboards)
# ---------------------------------------------------------------------------
set(NR_SHADER_DIR "${NR_CORE_DIR}/Backend/Vulkan/Shaders")
file(GLOB NR_RAYTRACER_SHADER_INCLUDES CONFIGURE_DEPENDS
    "${NR_SHADER_DIR}/Raytracer/*.slang")
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
    "${NR_SHADER_DIR}/Gaussian/Gaussian.slang")
set(NR_SHADER_OUT "${NR_SHADER_DIR}/Viewport/Viewport.spv" "${NR_SHADER_DIR}/Viewport/ViewportBillboards.spv"
    "${NR_SHADER_DIR}/Raytracer/Raytracer.spv"
    "${NR_SHADER_DIR}/Raytracer/RaytracingMiss.spv"
    "${NR_SHADER_DIR}/Raytracer/RaytracingHit.spv"
    "${NR_SHADER_DIR}/Raytracer/EmissionHit.spv"
    "${NR_SHADER_DIR}/Raytracer/OpacityAnyHit.spv"
    "${NR_SHADER_DIR}/Raytracer/GaussianAnyHit.spv"
    "${NR_SHADER_DIR}/Raytracer/GaussianHit.spv")
add_custom_command(OUTPUT ${NR_SHADER_OUT}
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Viewport/Viewport.slang"
        -entry main -target spirv -profile spirv_1_5+SPV_EXT_descriptor_heap
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -spirv-resource-heap-stride 128
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_DIR}/Viewport/Viewport.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Viewport/ViewportBillboards.slang"
        -entry vertMain -stage vertex -entry fragMain -stage fragment
        -target spirv -profile spirv_1_5+SPV_EXT_descriptor_heap
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -spirv-resource-heap-stride 128
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_DIR}/Viewport/ViewportBillboards.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/Raytracer.slang"
        -entry main -stage raygeneration -target spirv
        -profile spirv_1_5+spvRayTracingKHR+SPV_EXT_descriptor_heap
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -spirv-resource-heap-stride 128
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_DIR}/Raytracer/Raytracer.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/RaytracingMiss.slang"
        -entry main -stage miss -target spirv
        -profile spirv_1_5+spvRayTracingKHR+SPV_EXT_descriptor_heap
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -spirv-resource-heap-stride 128
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_DIR}/Raytracer/RaytracingMiss.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/RaytracingHit.slang"
        -entry main -stage closesthit -target spirv
        -profile spirv_1_5+spvRayTracingKHR+SPV_EXT_descriptor_heap
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -spirv-resource-heap-stride 128
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_DIR}/Raytracer/RaytracingHit.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/EmissionHit.slang"
        -entry main -stage closesthit -target spirv
        -profile spirv_1_5+spvRayTracingKHR+SPV_EXT_descriptor_heap
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -spirv-resource-heap-stride 128
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_DIR}/Raytracer/EmissionHit.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/OpacityAnyHit.slang"
        -entry main -stage anyhit -target spirv
        -profile spirv_1_5+spvRayTracingKHR+SPV_EXT_descriptor_heap
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -spirv-resource-heap-stride 128
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_DIR}/Raytracer/OpacityAnyHit.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/GaussianAnyHit.slang"
        -entry main -stage anyhit -target spirv
        -profile spirv_1_5+spvRayTracingKHR+SPV_EXT_descriptor_heap
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -spirv-resource-heap-stride 128
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_DIR}/Raytracer/GaussianAnyHit.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Raytracer/GaussianHit.slang"
        -entry main -stage closesthit -target spirv
        -profile spirv_1_5+spvRayTracingKHR+SPV_EXT_descriptor_heap
        -capability spvDescriptorHeapEXT -emit-spirv-directly
        -spirv-resource-heap-stride 128
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_DIR}/Raytracer/GaussianHit.spv"
    DEPENDS ${NR_SHADER_SRC} COMMENT "Compiling viewport shaders" VERBATIM)
add_custom_target(NoorRayShaders DEPENDS ${NR_SHADER_OUT})

# ---------------------------------------------------------------------------
# NoorRay (standalone application)
# ---------------------------------------------------------------------------
option(NR_BUILD_APPLICATION "Build the standalone NoorRay application" ON)
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
        NR_MATERIALX_STDLIB_DIR="${NR_MATERIALX_STDLIB_DIR}")
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(NoorRay PRIVATE -Wno-c23-extensions)
    endif()
endif()

# The Hydra delegate is an optional host plugin.  It is intentionally kept
# outside the default build because it needs Blender's ABI-matched OpenUSD
# SDK, but when enabled it links only against libnoorray and the gpu API.
option(NR_BUILD_HYDRA "Build the hdnoorray Hydra render delegate" OFF)
if(NR_BUILD_HYDRA)
    add_subdirectory(src/hdnoorray)
endif()

# ---------------------------------------------------------------------------
# _pynoorray (Python bindings)
# ---------------------------------------------------------------------------
option(NR_BUILD_PYTHON "Build Python bindings" ON)
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
option(NR_BUILD_TESTS "Build tests" ON)
if(NR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# gpu API examples
# ---------------------------------------------------------------------------
option(NR_BUILD_EXAMPLES "Build the standalone gpu API examples" ON)
if(NR_BUILD_EXAMPLES)
    add_subdirectory(examples/gpu_api)
endif()

# ---------------------------------------------------------------------------
# Link manifest for consumers that import libnoorray as a prebuilt archive
# ---------------------------------------------------------------------------
# A project that cannot add NoorRay as a subdirectory - because NoorRay defines
# global targets such as imgui and SDL3-static that would collide with its own -
# has to reproduce libnoorray's static dependency closure on its own link line.
# Doing that by hand means duplicating this file's knowledge in another
# repository, where it silently rots the moment a dependency is added here.
#
# Emit the closure instead. The archives are written in dependency order and
# wrapped in a linker group by the consumer, so the single-pass Unix linker does
# not care about the exact ordering within it.
set(NR_LINK_MANIFEST_TARGETS
    libnoorray noorray_gpu NoorRayVma NoorRayIO NoorRayMaterialX
    gauss_forge_readers tinyply tinyexr TBB::tbb spz
    webp sharpyuv zlibstatic
    MaterialXGenShader MaterialXFormat MaterialXCore)

set(NR_LINK_MANIFEST_ENTRIES "")
foreach(nr_target IN LISTS NR_LINK_MANIFEST_TARGETS)
    if(TARGET ${nr_target})
        # Interface targets have no archive of their own; their usage
        # requirements are already baked into the archives above.
        get_target_property(nr_type ${nr_target} TYPE)
        if(NOT nr_type STREQUAL "INTERFACE_LIBRARY")
            list(APPEND NR_LINK_MANIFEST_ENTRIES "$<TARGET_FILE:${nr_target}>")
        endif()
    endif()
endforeach()

file(GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/libnoorray-link.txt"
    CONTENT "$<JOIN:${NR_LINK_MANIFEST_ENTRIES},\n>\n")
