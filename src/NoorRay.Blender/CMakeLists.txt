#! The addon renders through the noorray module, which reads the depsgraph itself; the script next to it exports .nrs files with the same module.
set(addon_output "${CMAKE_BINARY_DIR}/NoorRay.Blender")
set(addon_package "${addon_output}/noorray_blender")
file(GLOB addon_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/noorray_blender/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/noorray_blender/*.toml")
set(export_script "${CMAKE_CURRENT_SOURCE_DIR}/export_scene.py")
if(PLATFORM_WINDOWS)
    set(addon_module noorray.pyd)
else()
    set(addon_module noorray.abi3.so)
endif()
add_custom_command(OUTPUT "${addon_output}/export_scene.py" "${addon_package}/${addon_module}"
                   COMMAND "${CMAKE_COMMAND}" -E rm -rf "${addon_output}"
                   COMMAND "${CMAKE_COMMAND}" -E make_directory "${addon_package}"
                   COMMAND "${CMAKE_COMMAND}" -E copy ${addon_sources} "${addon_package}"
                   COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:NoorRay.Python>" "${addon_package}/${addon_module}"
                   COMMAND "${CMAKE_COMMAND}" -E copy "${export_script}" "${addon_output}"
                   DEPENDS ${addon_sources} "${export_script}" NoorRay.Python
                   COMMENT "Staging the Blender addon and the export script"
                   VERBATIM)
add_custom_target(NoorRay.Blender ALL DEPENDS "${addon_output}/export_scene.py" "${addon_package}/${addon_module}")

#! The zip carries the module of each platform; the other one comes from the same preset's build for it, which a Linux build makes with llvm-mingw and a Windows build needs from Linux.
if(PLATFORM_LINUX)
    set(other_platform windows)
    set(other_module noorray.pyd)
else()
    set(other_platform linux)
    set(other_module noorray.abi3.so)
endif()
cmake_path(GET CMAKE_BINARY_DIR FILENAME build_name)
string(REGEX REPLACE "^build-(linux|windows)-" "build-${other_platform}-" other_build "${build_name}")
cmake_path(GET CMAKE_BINARY_DIR PARENT_PATH build_parent)
set(NOORRAY_ADDON_OTHER_MODULES "${build_parent}/${other_build}/src/NoorRay.Python/${other_module}"
    CACHE STRING "The NoorRay.Python modules of the other platforms for the Blender extension zip")
set(other_preset "")
if(PLATFORM_LINUX AND build_name MATCHES "^build-linux-" AND
   NOORRAY_ADDON_OTHER_MODULES STREQUAL "${build_parent}/${other_build}/src/NoorRay.Python/${other_module}")
    string(REGEX REPLACE "^build-" "" other_preset "${other_build}")
endif()
#! The zip takes the Linux module of the linux-portable-release preset, which needs only glibc 2.28 like Blender; a module of this build needs the glibc of the host.
set(portable_module "${build_parent}/build-linux-portable-release/src/NoorRay.Python/noorray.abi3.so")
set(own_module "$<TARGET_FILE:NoorRay.Python>")
set(own_target NoorRay.Python)
set(portable_script "")
if(PLATFORM_LINUX)
    set(NOORRAY_ADDON_LINUX_MODULE "${portable_module}" CACHE FILEPATH "The Linux module of the Blender extension zip")
    set(own_module "${NOORRAY_ADDON_LINUX_MODULE}")
    set(own_target "")
    if(NOORRAY_ADDON_LINUX_MODULE STREQUAL portable_module AND NOT build_name STREQUAL "build-linux-portable-release")
        set(portable_script "${PROJECT_SOURCE_DIR}/tools/build_portable_linux.sh")
    endif()
endif()
find_program(NOORRAY_LLVM_STRIP NAMES llvm-strip REQUIRED)
find_program(NOORRAY_BLENDER blender)
set(addon_zip "${CMAKE_BINARY_DIR}/noorray_blender-${PROJECT_VERSION}.zip")
add_custom_target(NoorRay.Blender.Zip
                  COMMAND "${CMAKE_COMMAND}" "-DSOURCES=${addon_sources}"
                          "-DMODULES=${own_module};${NOORRAY_ADDON_OTHER_MODULES}" "-DPORTABLE_SCRIPT=${portable_script}"
                          "-DSTAGE=${CMAKE_BINARY_DIR}/NoorRay.Blender.Zip" "-DOUTPUT=${addon_zip}"
                          "-DSTRIP=${NOORRAY_LLVM_STRIP}" "-DBLENDER=${NOORRAY_BLENDER}"
                          "-DOTHER_PRESET=${other_preset}" "-DOTHER_BUILD=${build_parent}/${other_build}"
                          "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}"
                          -P "${CMAKE_CURRENT_SOURCE_DIR}/PackageAddon.cmake"
                  DEPENDS ${own_target} ${addon_sources}
                  COMMENT "Packing the Blender extension for Linux and Windows"
                  VERBATIM)
