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
        # USD's headers pull in the full CPython API (pyconfig.h, patchlevel.h,
        # Python.h, ...), so vendor the whole include directory.
        file(RENAME "${_nr_usd_tmp}/python/include/python3.13"
             "${NR_VENDORED_USD_DIR}/include/python3.13")
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
# Blender's libusd_ms.so is built with GCC, which enables pxr's GNU
# hash_map/hash_set/SmallVector extensions on Linux (see
# pxr/base/arch/defines.h: ARCH_HAS_GNU_STL_EXTENSIONS requires
# ARCH_COMPILER_GCC). hdnoorray is a plugin loaded into Blender's own
# process, so it must instantiate those same pxr types -- e.g.
# TfHashMap's default allocator -- the same way, or symbols mangle
# differently and the plugin fails to load with an undefined symbol.
# Defining the macro here restores that code path when hdnoorray is
# built with clang, which libstdc++ (and its <ext/hash_map>) supports
# fine even though arch/defines.h only turns it on for GCC.
target_compile_definitions(NoorRayHydraUsd INTERFACE ARCH_HAS_GNU_STL_EXTENSIONS)
if(TARGET hd)
    target_link_libraries(NoorRayHydraUsd INTERFACE
            NoorRayUsd
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

add_library(hdnoorray MODULE
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
        imageLoader.cpp
        imageLoader.h
        lensMetadataBridge.cpp
        lensMetadataBridge.h
        material.cpp
        material.h
        memoryTextureRegistry.cpp
        memoryTextureRegistry.h
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

target_compile_features(hdnoorray PRIVATE cxx_std_23)
target_compile_definitions(hdnoorray PRIVATE HDNOORRAY_EXPORTS ARCH_HAS_GNU_STL_EXTENSIONS)
target_link_libraries(hdnoorray PRIVATE
        libnoorray
        NoorRayHydraUsd
        OpenGL::GL)
target_include_directories(hdnoorray PRIVATE
        "${PROJECT_SOURCE_DIR}/external/materialx/source")
target_compile_definitions(hdnoorray PRIVATE
        NR_MATERIALX_STDLIB_DIR="${NR_MATERIALX_STDLIB_DIR}")
if(UNIX AND NOT APPLE)
    # The Blender Flatpak does not provide CUDA's user-mode runtime. Bundle it
    # beside the delegate and let the dynamic loader resolve it locally.
    target_link_options(hdnoorray PRIVATE "LINKER:-rpath,$ORIGIN")
    # Blender and its USD/OIIO dependencies load their own TBB ABI, so
    # exporting the statically linked oneTBB runtime from this module would
    # make unversioned TBB symbols eligible for cross-DSO interposition with
    # Blender's own build. Localize it: calls originating in this module then
    # bind to the matching implementation, while the explicitly exported
    # Hydra plugin entry points remain public.
    target_link_options(hdnoorray PRIVATE
        "LINKER:--exclude-libs,libtbb.a")
endif()
set_target_properties(hdnoorray PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/hdnoorray/plugin")

set(HDNOORRAY_PLUGIN_LIBRARY
        "${CMAKE_SHARED_LIBRARY_PREFIX}hdnoorray${CMAKE_SHARED_LIBRARY_SUFFIX}")
configure_file(
        resources/plugInfo.json.in
        "${CMAKE_BINARY_DIR}/hdnoorray/plugin/plugInfo.json"
        @ONLY)

set(HDNOORRAY_BLENDER_EXTENSION_DIR
        "${CMAKE_BINARY_DIR}/hdnoorray/blender_extension/hdnoorray")
file(MAKE_DIRECTORY "${HDNOORRAY_BLENDER_EXTENSION_DIR}")

# Track Python files as proper dependencies so changes to them trigger a copy
# even when the C++ library doesn't need relinking.
add_custom_command(
    OUTPUT "${HDNOORRAY_BLENDER_EXTENSION_DIR}/__init__.py"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/blender_addon/hdnoorray/__init__.py"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/blender_addon/hdnoorray/__init__.py"
        "${HDNOORRAY_BLENDER_EXTENSION_DIR}/__init__.py"
    COMMENT "Copying __init__.py")
add_custom_command(
    OUTPUT "${HDNOORRAY_BLENDER_EXTENSION_DIR}/materialx_export.py"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/blender_addon/hdnoorray/materialx_export.py"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/blender_addon/hdnoorray/materialx_export.py"
        "${HDNOORRAY_BLENDER_EXTENSION_DIR}/materialx_export.py"
    COMMENT "Copying materialx_export.py")
add_custom_command(
    OUTPUT "${HDNOORRAY_BLENDER_EXTENSION_DIR}/materialx_sync.py"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/blender_addon/hdnoorray/materialx_sync.py"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/blender_addon/hdnoorray/materialx_sync.py"
        "${HDNOORRAY_BLENDER_EXTENSION_DIR}/materialx_sync.py"
    COMMENT "Copying materialx_sync.py")
add_custom_command(
    OUTPUT "${HDNOORRAY_BLENDER_EXTENSION_DIR}/sellmeier_node.py"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/blender_addon/hdnoorray/sellmeier_node.py"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/blender_addon/hdnoorray/sellmeier_node.py"
        "${HDNOORRAY_BLENDER_EXTENSION_DIR}/sellmeier_node.py"
    COMMENT "Copying sellmeier_node.py")
add_custom_command(
    OUTPUT "${HDNOORRAY_BLENDER_EXTENSION_DIR}/blender_manifest.toml"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/blender_addon/hdnoorray/blender_manifest.toml"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/blender_addon/hdnoorray/blender_manifest.toml"
        "${HDNOORRAY_BLENDER_EXTENSION_DIR}/blender_manifest.toml"
    COMMENT "Copying blender_manifest.toml")
add_custom_target(hdnoorray_py
    DEPENDS
        "${HDNOORRAY_BLENDER_EXTENSION_DIR}/__init__.py"
        "${HDNOORRAY_BLENDER_EXTENSION_DIR}/materialx_export.py"
        "${HDNOORRAY_BLENDER_EXTENSION_DIR}/materialx_sync.py"
        "${HDNOORRAY_BLENDER_EXTENSION_DIR}/sellmeier_node.py"
        "${HDNOORRAY_BLENDER_EXTENSION_DIR}/blender_manifest.toml")
add_dependencies(hdnoorray hdnoorray_py)

add_custom_command(TARGET hdnoorray POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${HDNOORRAY_BLENDER_EXTENSION_DIR}/plugin"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:hdnoorray>"
                "${CMAKE_BINARY_DIR}/hdnoorray/plugin/plugInfo.json"
                "${HDNOORRAY_BLENDER_EXTENSION_DIR}/plugin"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:CUDA::cudart>"
                "${HDNOORRAY_BLENDER_EXTENSION_DIR}/plugin/libcudart.so.13"
        COMMAND "${CMAKE_COMMAND}" -E tar cf
                "${CMAKE_BINARY_DIR}/hdnoorray/hdnoorray-blender-5.2.zip"
                --format=zip
                __init__.py
                materialx_export.py
                materialx_sync.py
                sellmeier_node.py
                blender_manifest.toml
                plugin
        WORKING_DIRECTORY "${HDNOORRAY_BLENDER_EXTENSION_DIR}"
        COMMENT "Assembling the installable hdnoorray Blender extension")

include(GNUInstallDirs)
install(TARGETS hdnoorray
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/usd/hdnoorray")
install(FILES "${CMAKE_BINARY_DIR}/hdnoorray/plugin/plugInfo.json"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/usd/hdnoorray")
