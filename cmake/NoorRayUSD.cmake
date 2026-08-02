# Shared OpenUSD setup used by the standalone scene I/O and the Hydra plugin.
# The Hydra subdirectory adds its more specific hd/plug/tf dependencies on top
# of this interface target when those targets are available.

set(NR_VENDORED_USD_DIR "${PROJECT_SOURCE_DIR}/external/blender-usd")

if(NR_USD_ROOT STREQUAL NR_VENDORED_USD_DIR)
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
        file(RENAME "${_nr_usd_tmp}/python/include/python3.13"
             "${NR_VENDORED_USD_DIR}/include/python3.13")
        file(REMOVE_RECURSE "${_nr_usd_tmp}")
    endif()

    if(NOT EXISTS "${NR_USD_ROOT}/lib/libusd_ms.so")
        find_library(_nr_usd_lib NAMES usd_ms
            HINTS "$ENV{BLENDER_LIB_DIR}" "/home/marcel/Programs/Blender/5.2/lib"
            PATHS /usr/lib/blender /app/lib NO_DEFAULT_PATH)
        if(_nr_usd_lib)
            file(MAKE_DIRECTORY "${NR_USD_ROOT}/lib")
            file(CREATE_LINK "${_nr_usd_lib}"
                 "${NR_USD_ROOT}/lib/libusd_ms.so" SYMBOLIC)
        else()
            message(FATAL_ERROR
                "Could not find libusd_ms.so. Install Blender 5.2, set BLENDER_LIB_DIR, or provide ${NR_USD_ROOT}/lib/libusd_ms.so")
        endif()
    endif()
else()
    if(NOT EXISTS "${NR_USD_ROOT}/include/pxr/pxr.h")
        message(FATAL_ERROR "NR_USD_ROOT set to '${NR_USD_ROOT}' but pxr/pxr.h not found")
    endif()
endif()

find_path(NR_USD_INCLUDE_DIR NAMES pxr/pxr.h
    PATHS "${NR_USD_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(NR_USD_MONOLITHIC_LIBRARY NAMES usd_ms
    PATHS "${NR_USD_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_library(NoorRayUsd INTERFACE)
# USD's vendored work headers inspect this macro before they include TBB's
# version header. NoorRay uses oneTBB (interface major 12), so make that
# choice explicit and avoid selecting USD's obsolete legacy-task branch.
target_compile_definitions(NoorRayUsd INTERFACE
    ARCH_HAS_GNU_STL_EXTENSIONS TBB_INTERFACE_VERSION_MAJOR=12)
target_include_directories(NoorRayUsd INTERFACE "${NR_USD_INCLUDE_DIR}")
target_link_libraries(NoorRayUsd INTERFACE "${NR_USD_MONOLITHIC_LIBRARY}")

# Blender intentionally ships USD with Python symbols unresolved because its
# host executable owns the Python runtime. Standalone NoorRay binaries need a
# small embedding runtime of their own; keep it in a separate target so the
# Hydra module still resolves Python from Blender, as intended.
get_filename_component(NR_USD_LIBRARY_DIR "${NR_USD_MONOLITHIC_LIBRARY}" REALPATH)
get_filename_component(NR_USD_LIBRARY_DIR "${NR_USD_LIBRARY_DIR}" DIRECTORY)
find_file(NR_BLENDER_PYTHON_LIBRARY NAMES libpython3.13.a
    PATHS "${NR_USD_ROOT}/../blender/5.2/python/lib"
          "/home/marcel/Programs/Blender/5.2/5.2/python/lib"
    NO_DEFAULT_PATH)
add_library(NoorRayUsdStandalone INTERFACE)
target_link_directories(NoorRayUsdStandalone INTERFACE "${NR_USD_LIBRARY_DIR}")
if(NR_BLENDER_PYTHON_LIBRARY)
    target_link_libraries(NoorRayUsdStandalone INTERFACE
        "${NR_BLENDER_PYTHON_LIBRARY}" dl pthread util m)
else()
    message(WARNING "Blender's static Python runtime was not found; standalone USD executables may need a host-provided Python runtime")
endif()
if(UNIX AND NOT APPLE)
    target_link_options(NoorRayUsdStandalone INTERFACE
        "LINKER:-rpath,${NR_USD_LIBRARY_DIR}")
endif()

find_path(NR_USD_PYTHON_INCLUDE_DIR NAMES pyconfig.h
    PATHS "${NR_USD_ROOT}/include/python3.13"
          "${NR_USD_ROOT}/../python/include/python3.13" NO_DEFAULT_PATH)
if(NR_USD_PYTHON_INCLUDE_DIR)
    target_include_directories(NoorRayUsd INTERFACE "${NR_USD_PYTHON_INCLUDE_DIR}")
endif()
find_path(NR_USD_PYTHON_PATCHLEVEL_DIR NAMES patchlevel.h
    PATHS /usr/include/python3.13)
if(NR_USD_PYTHON_PATCHLEVEL_DIR)
    target_include_directories(NoorRayUsd INTERFACE "${NR_USD_PYTHON_PATCHLEVEL_DIR}")
endif()
