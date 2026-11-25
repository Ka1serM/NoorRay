#! Packs the Blender extension zip of the addon sources and one stripped module per platform of the manifest; Blender validates it when installed here.
foreach(variable SOURCES MODULES STAGE OUTPUT STRIP)
    if(NOT DEFINED ${variable})
        message(FATAL_ERROR "PackageAddon.cmake needs -D${variable}=...")
    endif()
endforeach()

#! The module of the other platform, built through its preset in its own build folder.
if(OTHER_PRESET)
    if(NOT EXISTS "${OTHER_BUILD}/CMakeCache.txt")
        execute_process(COMMAND "${CMAKE_COMMAND}" --preset "${OTHER_PRESET}" WORKING_DIRECTORY "${SOURCE_DIR}"
                        COMMAND_ERROR_IS_FATAL ANY)
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" --build "${OTHER_BUILD}" --target NoorRay.Python COMMAND_ERROR_IS_FATAL ANY)
endif()

if(PORTABLE_SCRIPT)
    execute_process(COMMAND "${PORTABLE_SCRIPT}" COMMAND_ERROR_IS_FATAL ANY)
endif()

set(package "${STAGE}/noorray_blender")
file(REMOVE_RECURSE "${STAGE}")
file(MAKE_DIRECTORY "${package}")
file(COPY ${SOURCES} DESTINATION "${package}")

set(found_linux OFF)
set(found_windows OFF)
foreach(module IN LISTS MODULES)
    if(NOT EXISTS "${module}")
        message(FATAL_ERROR "The module ${module} is missing. Build NoorRay.Python in that build first, or set "
                            "NOORRAY_ADDON_OTHER_MODULES to the module of the other platform.")
    endif()
    cmake_path(GET module FILENAME name)
    if(name STREQUAL "noorray.abi3.so")
        set(found_linux ON)
    elseif(name STREQUAL "noorray.pyd")
        set(found_windows ON)
    else()
        message(FATAL_ERROR "${module} is neither noorray.abi3.so nor noorray.pyd")
    endif()
    execute_process(COMMAND "${STRIP}" --strip-all -o "${package}/${name}" "${module}" COMMAND_ERROR_IS_FATAL ANY)
endforeach()
if(NOT found_linux OR NOT found_windows)
    message(FATAL_ERROR "The zip needs noorray.abi3.so (linux-x64) and noorray.pyd (windows-x64); it has ${MODULES}")
endif()

#! Blender wants the manifest at the root of the zip.
file(REMOVE "${OUTPUT}")
file(GLOB entries RELATIVE "${package}" "${package}/*")
execute_process(COMMAND "${CMAKE_COMMAND}" -E tar cf "${OUTPUT}" --format=zip ${entries}
                WORKING_DIRECTORY "${package}" COMMAND_ERROR_IS_FATAL ANY)
if(BLENDER)
    execute_process(COMMAND "${BLENDER}" --factory-startup --command extension validate "${OUTPUT}"
                    COMMAND_ERROR_IS_FATAL ANY)
endif()
message(STATUS "Wrote ${OUTPUT}")
