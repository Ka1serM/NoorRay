cmake_minimum_required(VERSION 3.25)

if(NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "Set CMAKE_CXX_COMPILER")
endif()
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "" FORCE)
set(NR_CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA Toolkit 13.x root")
if(NOT EXISTS "${NR_CUDA_ROOT}/bin/nvcc")
    message(FATAL_ERROR "NR_CUDA_ROOT does not contain bin/nvcc")
endif()
set(CMAKE_CUDA_COMPILER "${NR_CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "" FORCE)
set(CUDAToolkit_ROOT "${NR_CUDA_ROOT}" CACHE PATH "" FORCE)
set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler" CACHE STRING "")
project(NoorRay VERSION 0.1.0 LANGUAGES C CXX CUDA)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "NoorRay requires Linux with NVIDIA CUDA/OptiX")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_SKIP_BUILD_RPATH ON)

set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
set(NR_CUDA_ARCH "86;89" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_ARCHITECTURES "86;89" CACHE STRING "" FORCE)
list(GET NR_CUDA_ARCH 0 NR_OPTIX_CUDA_ARCH)

find_package(CUDAToolkit 13 REQUIRED)
if(NOT CUDAToolkit_VERSION_MAJOR EQUAL 13)
    message(FATAL_ERROR "NoorRay requires CUDA Toolkit 13.x")
endif()
set(OPTIX_ROOT "/home/marcel/Programs/OptixSDK" CACHE PATH "OptiX 9.1 installation root")
find_path(OPTIX_INCLUDE_DIR optix.h PATHS "${OPTIX_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
file(STRINGS "${OPTIX_INCLUDE_DIR}/optix.h" OPTIX_VERSION_LINE REGEX "^#define OPTIX_VERSION 90100$")
if(NOT OPTIX_VERSION_LINE)
    message(FATAL_ERROR "NoorRay requires OptiX 9.1 headers")
endif()

set(NR_CORE_DIR "${PROJECT_SOURCE_DIR}/src/libnoorray")
set(NR_APP_DIR "${PROJECT_SOURCE_DIR}/src/app")
set(NR_PTX_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/build/generated")

set(NR_USD_ROOT "${PROJECT_SOURCE_DIR}/external/blender-usd" CACHE PATH
    "Blender USD SDK root (auto-vendored by default)")
include(cmake/NoorRayUSD.cmake)

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

include(cmake/ROSS.cmake)

# ---------------------------------------------------------------------------
# GaussForge (gaussian splat readers)
# ---------------------------------------------------------------------------
include(FetchContent)
include(external/GaussForge/cmakes/zlib.cmake)
if(TARGET zlibstatic)
    set_target_properties(zlibstatic PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
FetchContent_Declare(libwebp
    GIT_REPOSITORY https://github.com/webmproject/libwebp.git
    GIT_TAG v1.4.0 GIT_SHALLOW TRUE)
foreach(_v WEBP_BUILD_ANIM_UTILS WEBP_BUILD_CWEBP WEBP_BUILD_DWEBP WEBP_BUILD_GIF2WEBP
         WEBP_BUILD_IMG2WEBP WEBP_BUILD_VWEBP WEBP_BUILD_WEBPINFO WEBP_BUILD_LIBWEBPMUX
         WEBP_BUILD_WEBPMUX WEBP_BUILD_EXTRAS)
    set(${_v} OFF CACHE BOOL "" FORCE)
endforeach()
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
    external/GaussForge/src/io/sog_writer.cpp)
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
# PTX compilation (OptiX modules)
# ---------------------------------------------------------------------------
set(NR_OPTIX_PTX_OPTIONS
    -ccbin /usr/bin/g++-15 --ptx "--gpu-architecture=compute_${NR_OPTIX_CUDA_ARCH}"
    --std=c++20 --use_fast_math --expt-relaxed-constexpr --extended-lambda
    -allow-unsupported-compiler -DFMT_USE_BITINT=0)
set(NR_OPTIX_INC
    "-I${NR_CORE_DIR}" "-I${NR_APP_DIR}" "-I${PROJECT_SOURCE_DIR}/external/glm" "-I${OPTIX_INCLUDE_DIR}"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/libross/main"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/ext/nlohmann-json/include"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/ext/spdlog/include"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/ext/fmt/include"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/ext/magic_enum/include"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/openlensfileio/src/openlensfileio/main"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/ext/tabulate/include"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/ext/libmorton/include"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/ext/gilbert"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/ext/ordered_map/include"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/ext/lodepng"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/ext/hash-library"
    "-I${PROJECT_SOURCE_DIR}/external/tbb/include"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/pbrt-v4/src/ext/openexr/src/lib/OpenEXR"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/pbrt-v4/src/ext/openexr/src/lib/Iex"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/pbrt-v4/src/ext/openexr/src/lib/IlmThread"
    "-I${PROJECT_SOURCE_DIR}/external/ROSS/pbrt-v4/src/ext/openexr/src/lib/OpenEXRCore"
    "-I${CMAKE_CURRENT_BINARY_DIR}/ross/openexr/cmake"
    "-I${CMAKE_CURRENT_BINARY_DIR}/_deps/imath-src/src/Imath"
    "-I${CMAKE_CURRENT_BINARY_DIR}/_deps/imath-build/config")

function(nr_ptx name source)
    set(ptx "${NR_PTX_OUTPUT_DIR}/${name}.ptx")
    set(dep "${NR_PTX_OUTPUT_DIR}/${name}.d")
    add_custom_command(OUTPUT "${ptx}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NR_PTX_OUTPUT_DIR}"
        COMMAND "${CMAKE_CUDA_COMPILER}" ${NR_OPTIX_PTX_OPTIONS}
            -MD --dependency-output "${dep}" -DNR_BACKEND_CUDA=1 ${ARGN}
            -o "${ptx}" -x cu "${source}"
        DEPENDS "${source}" DEPFILE "${dep}" VERBATIM)
    add_custom_target("${name}" DEPENDS "${ptx}")
endfunction()

nr_ptx(NoorRayOptixIr "${NR_CORE_DIR}/Backend/OptiX/Kernels/Generate.h"
    -DROSS_BUILD_GPU_SUPPORT=1 ${NR_OPTIX_INC})
nr_ptx(NoorRayOptixMissIr "${NR_CORE_DIR}/Backend/OptiX/Kernels/Miss.h"
    ${NR_OPTIX_INC})
# ---------------------------------------------------------------------------
# NoorRayCuda (CUDA / OptiX pipeline library)
# ---------------------------------------------------------------------------
set(NR_CUDA_SOURCES
    src/libnoorray/Backend/OptiX/Acceleration/Blas.cpp
    src/libnoorray/Backend/OptiX/Acceleration/AnalyticLightBlas.cpp
    src/libnoorray/Backend/OptiX/Acceleration/GaussianProxyBlas.cpp
    src/libnoorray/Backend/OptiX/Acceleration/Tlas.cpp
    src/libnoorray/Backend/OptiX/Denoising/OptixDenoiser.cpp
    src/libnoorray/Backend/OptiX/Runtime/Raytracer.cpp
    src/libnoorray/Materials/SVM/SvmProgramTable.cpp
    src/libnoorray/Backend/CUDA/Unique/Texture.cpp
    src/libnoorray/Backend/CUDA/Unique/SharedImage.cpp
    src/libnoorray/Backend/CUDA/Unique/SharedBuffer.cpp
    src/libnoorray/Backend/CUDA/rstd/Memory.cpp
    src/libnoorray/Materials/Shading/EnergyLut.cpp
    src/libnoorray/Materials/Shading/RgbSpectrumTableData.cpp)

add_library(NoorRayCuda STATIC ${NR_CUDA_SOURCES})
add_dependencies(NoorRayCuda NoorRayOptixIr NoorRayOptixMissIr)
set_target_properties(NoorRayCuda PROPERTIES
    CUDA_ARCHITECTURES "${NR_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(NoorRayCuda PUBLIC cxx_std_20 cuda_std_20)
target_compile_definitions(NoorRayCuda PUBLIC NR_BACKEND_CUDA=1 VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
target_include_directories(NoorRayCuda PUBLIC
    "${NR_CORE_DIR}" "${NR_APP_DIR}" "${PROJECT_SOURCE_DIR}/external/glm"
    "${PROJECT_SOURCE_DIR}/external/SDL/include"
    "${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include"
    "${PROJECT_SOURCE_DIR}/external/portable-file-dialogs"
    "${Vulkan_INCLUDE_DIR}" "${OPTIX_INCLUDE_DIR}")
target_compile_options(NoorRayCuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
target_link_libraries(NoorRayCuda PUBLIC
    CUDA::cudart_static CUDA::cuda_driver CUDA::nvtx3 glm stb
    Vulkan::Vulkan portable_file_dialogs GPUOpen::VulkanMemoryAllocator gauss_forge_readers libross)

# ---------------------------------------------------------------------------
# libnoorray (core engine library)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE NR_CORE_FILES CONFIGURE_DEPENDS
    "${NR_CORE_DIR}/*.cpp" "${NR_CORE_DIR}/*.h")
set(NR_CORE_X
    "/Backend/CUDA/" "/Backend/OptiX/Kernels/" "/Materials/MaterialX/MaterialXDocument\\.cpp$"
    "/Materials/Shading/EnergyLut/GenerateEnergyLuts\\.cpp$" "/Materials/Shading/EnergyLut\\.cpp$"
    "/Backend/OptiX/Runtime/Raytracer\\.cpp$"
    "/Materials/Shading/RgbSpectrumTableData\\.cpp$"
    "/Backend/OptiX/Acceleration/" "/Backend/OptiX/Denoising/"
    "/IO/BitmapReader\\.cpp$" "/IO/BitmapWriter\\.cpp$" "/StbImpl\\.cpp$"
    "/Materials/SVM/references/" "/Materials/SVM/SvmProgramTable\\.cpp$")
foreach(p IN LISTS NR_CORE_X)
    list(FILTER NR_CORE_FILES EXCLUDE REGEX "${p}")
endforeach()
add_library(libnoorray STATIC ${NR_CORE_FILES})
add_dependencies(libnoorray NoorRayShaders)
set_target_properties(libnoorray PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    POSITION_INDEPENDENT_CODE ON)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(libnoorray PRIVATE -Wno-c23-extensions)
endif()
target_compile_features(libnoorray PUBLIC cxx_std_23)
target_include_directories(libnoorray PUBLIC "${NR_CORE_DIR}")
target_compile_definitions(libnoorray PUBLIC
    NR_BACKEND_CUDA=1 VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    NOORRAY_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets"
    NR_ROSS_RESOURCE_DIR="${PROJECT_SOURCE_DIR}/external/ROSS/resources"
    NR_OPTIX_CUDA_ARCH=\"${NR_OPTIX_CUDA_ARCH}\")
target_link_libraries(libnoorray PUBLIC
    NoorRayCuda NoorRayIO portable_file_dialogs glm Vulkan::Vulkan
    tinygltf tinyobjloader tinyply tinyexr stb gauss_forge_readers
    GPUOpen::VulkanMemoryAllocator glaze TBB::tbb
    NoorRayMaterialX MaterialXFormat MaterialXCore NoorRayUsd)

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
target_include_directories(NoorRayMaterialX PUBLIC "${NR_CORE_DIR}" PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(NoorRayMaterialX PUBLIC
    glm MaterialXGenShader MaterialXFormat MaterialXCore CUDA::cudart_static)
target_compile_definitions(NoorRayMaterialX PRIVATE
    NR_MATERIALX_STDLIB_DIR="${NR_MATERIALX_STDLIB_DIR}")
set_target_properties(NoorRayMaterialX PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Vulkan shaders (Viewport, ViewportBillboards)
# ---------------------------------------------------------------------------
set(NR_SHADER_DIR "${NR_CORE_DIR}/Backend/Vulkan/Shaders")
set(NR_SHADER_SRC "${NR_SHADER_DIR}/Viewport/Viewport.slang" "${NR_SHADER_DIR}/Viewport/ViewportBillboards.slang")
set(NR_SHADER_OUT "${NR_SHADER_DIR}/Viewport/Viewport.spv" "${NR_SHADER_DIR}/Viewport/ViewportBillboards.spv")
add_custom_command(OUTPUT ${NR_SHADER_OUT}
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Viewport/Viewport.slang"
        -entry main -target spirv -profile spirv_1_5 -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_DIR}/Viewport/Viewport.spv"
    COMMAND "${NR_SLANGC_EXECUTABLE}" "${NR_SHADER_DIR}/Viewport/ViewportBillboards.slang"
        -entry vertMain -stage vertex -entry fragMain -stage fragment
        -target spirv -profile spirv_1_5 -emit-spirv-directly
        -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081
        -o "${NR_SHADER_DIR}/Viewport/ViewportBillboards.spv"
    DEPENDS ${NR_SHADER_SRC} COMMENT "Compiling viewport shaders" VERBATIM)
add_custom_target(NoorRayShaders DEPENDS ${NR_SHADER_OUT})

# ---------------------------------------------------------------------------
# hdnoorray (Blender Hydra render delegate)
# ---------------------------------------------------------------------------
add_subdirectory(src/hdnoorray)

# ---------------------------------------------------------------------------
# NoorRay (standalone application)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE NR_APP_FILES CONFIGURE_DEPENDS "${NR_APP_DIR}/*.cpp" "${NR_APP_DIR}/*.h")
add_executable(NoorRay ${NR_APP_FILES})
set_target_properties(NoorRay PROPERTIES
    PREFIX "" CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_dependencies(NoorRay NoorRayShaders)
target_include_directories(NoorRay PRIVATE
    "${NR_APP_DIR}" "${NR_CORE_DIR}" "${PROJECT_SOURCE_DIR}/external/materialx/source")
target_link_libraries(NoorRay PRIVATE
    libnoorray NoorRayCuda libross portable_file_dialogs glm imgui imnodeflow Vulkan::Vulkan
    SDL3::SDL3-static tinygltf tinyobjloader tinyexr stb gauss_forge_readers
    ImViewGuizmo GPUOpen::VulkanMemoryAllocator glaze TBB::tbb NoorRayMaterialX
    NoorRayUsdStandalone)
target_compile_definitions(NoorRay PRIVATE
    SDL_MAIN_HANDLED NR_BACKEND_CUDA=1 VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    NR_MATERIALX_STDLIB_DIR="${NR_MATERIALX_STDLIB_DIR}")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(NoorRay PRIVATE -Wno-c23-extensions)
endif()
target_link_options(NoorRay PRIVATE -static-libstdc++)

# ---------------------------------------------------------------------------
# _pynoorray (Python bindings)
# ---------------------------------------------------------------------------
option(NR_BUILD_PYTHON "Build Python bindings" ON)
if(NR_BUILD_PYTHON)
    set(Python_EXECUTABLE "/usr/bin/python3.13" CACHE FILEPATH "Python interpreter")
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
        "$<LINK_GROUP:RESCAN,libnoorray,NoorRayCuda,NoorRayIO,libross>"
        CUDA::cudart_static)
    target_compile_definitions(_pynoorray PRIVATE
        NR_BACKEND_CUDA=1 VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
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
