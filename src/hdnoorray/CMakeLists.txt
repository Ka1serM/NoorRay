cmake_minimum_required(VERSION 3.25)

# --------------------------------------------------------------------------
# Automatic setup of the vendored Blender USD SDK
# --------------------------------------------------------------------------
set(NR_VENDORED_USD_DIR "${PROJECT_SOURCE_DIR}/external/blender-usd")

if(NR_USD_ROOT STREQUAL NR_VENDORED_USD_DIR)
    # Vendored headers — fetch from Blender's dependency package if missing.
    if(NOT EXISTS "${NR_USD_ROOT}/include/pxr/pxr.h")
        message(STATUS "Vendoring USD headers from Blender 5.2 dependency package...")
        find_package(Git REQUIRED)
        set(_nr_usd_tmp "${NR_VENDORED_USD_DIR}/.tmp")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" clone --depth 1 --filter=blob:none --sparse
                    --branch blender-v5.2-release
                    https://projects.blender.org/blender/lib-linux_x64.git
                    "${_nr_usd_tmp}"
            COMMAND_ERROR_IS_FATAL ANY)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${_nr_usd_tmp}"
                    sparse-checkout set usd/include python/include
            COMMAND_ERROR_IS_FATAL ANY)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${_nr_usd_tmp}" checkout
            COMMAND_ERROR_IS_FATAL ANY)
        file(MAKE_DIRECTORY "${NR_VENDORED_USD_DIR}/include")
        file(RENAME "${_nr_usd_tmp}/usd/include/pxr"
             "${NR_VENDORED_USD_DIR}/include/pxr")
        file(MAKE_DIRECTORY "${NR_VENDORED_USD_DIR}/include/python3.13")
        file(RENAME "${_nr_usd_tmp}/python/include/python3.13/pyconfig.h"
             "${NR_VENDORED_USD_DIR}/include/python3.13/pyconfig.h")
        file(REMOVE_RECURSE "${_nr_usd_tmp}")
        message(STATUS "Vendored USD headers at ${NR_VENDORED_USD_DIR}")
    endif()

    # Vendored library — search common Blender install paths when missing.
    if(NOT EXISTS "${NR_USD_ROOT}/lib/libusd_ms.so")
        find_library(_nr_usd_lib
            NAMES usd_ms
            HINTS
                "$ENV{BLENDER_LIB_DIR}"
                "/home/marcel/Programs/Blender/5.2/lib"
            PATHS
                /usr/lib/blender
                /app/lib
            NO_DEFAULT_PATH)
        if(_nr_usd_lib)
            file(MAKE_DIRECTORY "${NR_USD_ROOT}/lib")
            file(CREATE_LINK "${_nr_usd_lib}"
                 "${NR_USD_ROOT}/lib/libusd_ms.so" SYMBOLIC)
            message(STATUS "Linked libusd_ms.so from ${_nr_usd_lib}")
        else()
            message(FATAL_ERROR
                "Could not find libusd_ms.so. Install Blender 5.2, set "
                "BLENDER_LIB_DIR, or symlink manually:\n"
                "  ln -s /path/to/blender/5.2/lib/libusd_ms.so "
                "${NR_VENDORED_USD_DIR}/lib/libusd_ms.so")
        endif()
    endif()
else()
    # Custom NR_USD_ROOT — user is responsible for providing the SDK.
    if(NOT EXISTS "${NR_USD_ROOT}/include/pxr/pxr.h")
        message(FATAL_ERROR
                "NR_USD_ROOT set to '${NR_USD_ROOT}' but pxr/pxr.h not found. "
                "Install Blender 5.2 or vendor the SDK at "
                "${NR_VENDORED_USD_DIR}.")
    endif()
endif()

# A normal OpenUSD installation exports individual CMake targets. Blender
# instead ships a monolithic libusd_ms, so support both layouts.
find_package(pxr CONFIG QUIET
        PATHS "${NR_USD_ROOT}"
        NO_DEFAULT_PATH)

add_library(NoorRayHydraUsd INTERFACE)
if(TARGET hd)
    target_link_libraries(NoorRayHydraUsd INTERFACE
            hd
            plug
            tf
            usdLux)
else()
    find_path(NR_USD_INCLUDE_DIR
            NAMES pxr/pxr.h
            PATHS "${NR_USD_ROOT}/include"
            NO_DEFAULT_PATH
            REQUIRED)
    find_library(NR_USD_MONOLITHIC_LIBRARY
            NAMES usd_ms
            PATHS "${NR_USD_ROOT}/lib"
            NO_DEFAULT_PATH
            REQUIRED)
    target_include_directories(NoorRayHydraUsd INTERFACE "${NR_USD_INCLUDE_DIR}")
    target_link_libraries(NoorRayHydraUsd INTERFACE "${NR_USD_MONOLITHIC_LIBRARY}")
endif()

find_package(OpenGL REQUIRED)

# Blender's OpenUSD build has Python support enabled, so even non-Python Hydra
# headers transitively include pyconfig.h. Its dependency package places the
# matching Python SDK beside the usd directory.
find_path(NR_USD_PYTHON_INCLUDE_DIR
        NAMES pyconfig.h
        PATHS
            "${NR_USD_ROOT}/include/python3.13"
            "${NR_USD_ROOT}/../python/include/python3.13"
        NO_DEFAULT_PATH)
if(NR_USD_PYTHON_INCLUDE_DIR)
    target_include_directories(
            NoorRayHydraUsd INTERFACE "${NR_USD_PYTHON_INCLUDE_DIR}")
endif()

# The vendored Python include only provides pyconfig.h, but USD headers also
# need patchlevel.h. Find the system Python include dir to supply the rest.
find_path(NR_USD_PYTHON_PATCHLEVEL_DIR
        NAMES patchlevel.h
        PATHS /usr/include/python3.13)
if(NR_USD_PYTHON_PATCHLEVEL_DIR)
    target_include_directories(
            NoorRayHydraUsd INTERFACE "${NR_USD_PYTHON_PATCHLEVEL_DIR}")
endif()

add_library(hdNoorRay MODULE
        api.h
        instancer.cpp
        instancer.h
        lights/diskLight.cpp
        lights/diskLight.h
        lights/distantLight.cpp
        lights/distantLight.h
        lights/domeLight.cpp
        lights/domeLight.h
        lights/light.cpp
        lights/light.h
        lights/rectLight.cpp
        lights/rectLight.h
        lights/sphereLight.cpp
        lights/sphereLight.h
        material.cpp
        material.h
        mesh.cpp
        mesh.h
        renderDelegate.cpp
        renderDelegate.h
        renderBuffer.cpp
        renderBuffer.h
        renderParam.cpp
        renderParam.h
        renderPass.cpp
        renderPass.h
        rendererPlugin.cpp
        rendererPlugin.h)

target_compile_features(hdNoorRay PRIVATE cxx_std_23)
target_compile_definitions(hdNoorRay PRIVATE HDNOORRAY_EXPORTS)
target_link_libraries(hdNoorRay PRIVATE
        libnoorray
        NoorRayHydraUsd
        OpenGL::GL)
if(UNIX AND NOT APPLE)
    # The Blender Flatpak does not provide CUDA's user-mode runtime. Bundle it
    # beside the delegate and let the dynamic loader resolve it locally.
    target_link_options(hdNoorRay PRIVATE "LINKER:-rpath,$ORIGIN")
endif()
set_target_properties(hdNoorRay PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/hdnoorray/plugin")

set(HDNOORRAY_PLUGIN_LIBRARY
        "${CMAKE_SHARED_LIBRARY_PREFIX}hdNoorRay${CMAKE_SHARED_LIBRARY_SUFFIX}")
configure_file(
        resources/plugInfo.json.in
        "${CMAKE_BINARY_DIR}/hdnoorray/plugin/plugInfo.json"
        @ONLY)

set(HDNOORRAY_BLENDER_EXTENSION_DIR
        "${CMAKE_BINARY_DIR}/hdnoorray/blender_extension/hdnoorray")
file(MAKE_DIRECTORY "${HDNOORRAY_BLENDER_EXTENSION_DIR}")
add_custom_command(TARGET hdNoorRay POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${HDNOORRAY_BLENDER_EXTENSION_DIR}/plugin"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/blender_addon/hdnoorray/__init__.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/blender_addon/hdnoorray/blender_manifest.toml"
                "${HDNOORRAY_BLENDER_EXTENSION_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:hdNoorRay>"
                "${CMAKE_BINARY_DIR}/hdnoorray/plugin/plugInfo.json"
                "${HDNOORRAY_BLENDER_EXTENSION_DIR}/plugin"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:CUDA::cudart>"
                "${HDNOORRAY_BLENDER_EXTENSION_DIR}/plugin/libcudart.so.13"
        COMMAND "${CMAKE_COMMAND}" -E tar cf
                "${CMAKE_BINARY_DIR}/hdnoorray/hdnoorray-blender-5.2.zip"
                --format=zip
                __init__.py
                blender_manifest.toml
                plugin
        WORKING_DIRECTORY "${HDNOORRAY_BLENDER_EXTENSION_DIR}"
        COMMENT "Assembling the installable hdNoorRay Blender extension")

include(GNUInstallDirs)
install(TARGETS hdNoorRay
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/usd/hdNoorRay")
install(FILES "${CMAKE_BINARY_DIR}/hdnoorray/plugin/plugInfo.json"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/usd/hdNoorRay")
