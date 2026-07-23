cmake_minimum_required(VERSION 3.25)

if(NOT NR_USD_ROOT)
    message(FATAL_ERROR
            "NR_BUILD_HYDRA requires NR_USD_ROOT to point at an OpenUSD SDK. "
            "For Blender, use the headers and libusd_ms from the Blender 5.2 dependency package.")
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
            tf)
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

add_library(hdNoorRay MODULE
        api.h
        instancer.cpp
        instancer.h
        light.cpp
        light.h
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
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/hdnoorray/blender_extension")
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
