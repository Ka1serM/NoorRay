# The gpu library is self-contained: Vulkan, Vulkan-HPP and VMA are its only
# dependencies. It knows nothing about NoorRay, shader compilers or assets.
#
# This file works two ways: as add_subdirectory() from a host project, and as
# the top-level CMakeLists of a standalone build. Everything that only makes
# sense for the standalone case is guarded by GPU_IS_TOP_LEVEL.
cmake_minimum_required(VERSION 3.25)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(gpu VERSION 0.1.0 LANGUAGES CXX)
    set(GPU_IS_TOP_LEVEL ON)
    set(GPU_VERSION "${PROJECT_VERSION}")
else()
    set(GPU_IS_TOP_LEVEL OFF)
    # Not a project() here, so PROJECT_VERSION belongs to the host. Keep our own.
    set(GPU_VERSION "0.1.0")
endif()

option(GPU_BUILD_EXAMPLES "Build the gpu API examples" ${GPU_IS_TOP_LEVEL})
option(GPU_BUILD_TESTS "Build the gpu API tests" ${GPU_IS_TOP_LEVEL})

if(GPU_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

include(GNUInstallDirs)

find_package(Vulkan REQUIRED)

# VMA ships only headers plus one implementation TU, which is ours to compile
# (src/vma_impl.cpp). Prefer an installed package; fall back to the vendored
# copy so a fresh clone with submodules configures without a system VMA.
if(NOT TARGET GPUOpen::VulkanMemoryAllocator)
    find_package(VulkanMemoryAllocator QUIET)
endif()
if(NOT TARGET GPUOpen::VulkanMemoryAllocator)
    # The vendored location once this is its own repository, then the copy in
    # the surrounding NoorRay tree so a standalone configure works from here
    # too, before the split.
    set(GPU_VMA_DIR "" CACHE PATH "VulkanMemoryAllocator source directory")
    foreach(_candidate
            "${GPU_VMA_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/external/VulkanMemoryAllocator"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../external/VulkanMemoryAllocator")
        if(_candidate AND EXISTS "${_candidate}/CMakeLists.txt")
            add_subdirectory("${_candidate}"
                "${CMAKE_CURRENT_BINARY_DIR}/VulkanMemoryAllocator" EXCLUDE_FROM_ALL)
            break()
        endif()
    endforeach()
endif()
if(NOT TARGET GPUOpen::VulkanMemoryAllocator)
    message(FATAL_ERROR
        "VulkanMemoryAllocator not found. Install it, set GPU_VMA_DIR, or "
        "populate ${CMAKE_CURRENT_SOURCE_DIR}/external/VulkanMemoryAllocator.")
endif()

add_library(gpu STATIC
    src/device.cpp
    src/graphics.cpp
    src/swapchain.cpp
    src/image.cpp
    src/vma_impl.cpp
    src/internal.hpp)
add_library(gpu::gpu ALIAS gpu)
# Transitional alias for hosts still linking the old target name.
add_library(noorray_gpu ALIAS gpu)

set_target_properties(gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(gpu PUBLIC cxx_std_23)

# Only the public headers are visible to consumers; src/ holds internal.hpp,
# which pulls in Vulkan-HPP and VMA and must not leak into dependents.
target_include_directories(gpu
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
           "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_compile_definitions(gpu PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
# VMA is a private implementation detail: it appears in src/internal.hpp and
# src/vma_impl.cpp only, never in include/gpu. It is header-only, so nothing of
# it survives into the archive's usage requirements - BUILD_INTERFACE keeps it
# out of the export set, which would otherwise demand VMA be exported too.
target_link_libraries(gpu
    PUBLIC Vulkan::Vulkan
    PRIVATE $<BUILD_INTERFACE:GPUOpen::VulkanMemoryAllocator>)

if(GPU_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(GPU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Install / export
# ---------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

install(TARGETS gpu
    EXPORT gpuTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

install(DIRECTORY include/gpu DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(EXPORT gpuTargets
    FILE gpuTargets.cmake
    NAMESPACE gpu::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/gpu")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpuConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/gpuConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/gpu")

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/gpuConfigVersion.cmake"
    VERSION ${GPU_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/gpuConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/gpuConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/gpu")
