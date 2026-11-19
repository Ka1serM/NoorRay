#! Third-party projects compile without warnings: they are not ours to fix.
set(no_warnings -w)
function(add_external_subdirectory source)
    get_directory_property(options COMPILE_OPTIONS)
    add_compile_options(${no_warnings})
    add_subdirectory(${source} ${ARGN} EXCLUDE_FROM_ALL SYSTEM)
    set_directory_properties(PROPERTIES COMPILE_OPTIONS "${options}")
endfunction()

set(submodules NoorRHI RTXDI flatbuffers stb tinyexr tinygltf tinyobjloader)
if(NOORRAY_BUILD_APP)
    list(APPEND submodules RmlKit)
endif()
foreach(submodule ${submodules})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${submodule}/.git")
        message(FATAL_ERROR "external/${submodule} is missing. Run: git submodule update --init --recursive")
    endif()
endforeach()

add_subdirectory(NoorRHI EXCLUDE_FROM_ALL)

if(NOORRAY_BUILD_APP)
    add_subdirectory(RmlKit EXCLUDE_FROM_ALL)
endif()

#! A cross build (the browser, or Windows from Linux) builds flatc with the Clang of the build machine; it compares the systems because the Windows preset sets CMAKE_SYSTEM_NAME on a Windows host too.
foreach(flatbuffers_option BUILD_TESTS INSTALL BUILD_FLATLIB)
    set(FLATBUFFERS_${flatbuffers_option} OFF CACHE BOOL "" FORCE)
endforeach()
if(NOT CMAKE_SYSTEM_NAME STREQUAL CMAKE_HOST_SYSTEM_NAME)
    include(ExternalProject)
    set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "" FORCE)
    set(host_flatc_dir "${CMAKE_CURRENT_BINARY_DIR}/host-flatc")
    set(FLATC_COMMAND "${host_flatc_dir}/flatc${CMAKE_HOST_EXECUTABLE_SUFFIX}" CACHE INTERNAL "")
    ExternalProject_Add(host_flatc
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers"
        BINARY_DIR "${host_flatc_dir}"
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++ "-DCMAKE_CXX_FLAGS=-O0 -stdlib=libc++"
            "-DCMAKE_EXE_LINKER_FLAGS=-stdlib=libc++ --rtlib=compiler-rt --unwindlib=libunwind"
            -DFLATBUFFERS_BUILD_TESTS=OFF -DFLATBUFFERS_BUILD_FLATLIB=OFF -DFLATBUFFERS_INSTALL=OFF
        BUILD_COMMAND "${CMAKE_COMMAND}" --build "${host_flatc_dir}" --target flatc
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS "${host_flatc_dir}/flatc${CMAKE_HOST_EXECUTABLE_SUFFIX}")
    set(FLATC_DEPENDENCY host_flatc CACHE INTERNAL "")
else()
    set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
    set(FLATC_COMMAND "$<TARGET_FILE:flatc>" CACHE INTERNAL "")
    set(FLATC_DEPENDENCY flatc CACHE INTERNAL "")
endif()
add_external_subdirectory(flatbuffers)
if(TARGET flatc)
    target_compile_options(flatc PRIVATE -O0)
    set_target_properties(flatc PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

add_library(stb INTERFACE)
target_include_directories(stb SYSTEM INTERFACE stb)

add_library(tinygltf INTERFACE)
target_include_directories(tinygltf SYSTEM INTERFACE tinygltf)
target_compile_definitions(tinygltf INTERFACE TINYGLTF_NOEXCEPTION JSON_NOEXCEPTION TINYGLTF_NO_STB_IMAGE TINYGLTF_NO_STB_IMAGE_WRITE TINYGLTF_NO_INCLUDE_STB_IMAGE TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE)

add_library(tinyobjloader INTERFACE)
target_include_directories(tinyobjloader SYSTEM INTERFACE tinyobjloader)

add_library(tinyexr STATIC tinyexr/tinyexr.cc tinyexr/deps/miniz/miniz.c)
target_include_directories(tinyexr SYSTEM PUBLIC tinyexr tinyexr/deps/miniz)
target_compile_definitions(tinyexr PUBLIC TINYEXR_USE_MINIZ=1 TINYEXR_USE_THREAD=1)
target_compile_options(tinyexr PRIVATE ${no_warnings})

add_library(rtxdi STATIC RTXDI/Source/ReSTIRPT.cpp RTXDI/Source/RtxdiUtils.cpp)
target_include_directories(rtxdi SYSTEM PUBLIC RTXDI/Include)
target_compile_definitions(rtxdi PUBLIC RTXDI_COLOR_COMPONENTS=4)
target_compile_features(rtxdi PRIVATE cxx_std_20)
target_compile_options(rtxdi PRIVATE ${no_warnings})

if(NOORRAY_USD_ROOT)
    find_path(NOORRAY_USD_INCLUDE_DIR NAMES pxr/pxr.h PATHS "${NOORRAY_USD_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
    find_library(NOORRAY_USD_LIBRARY NAMES usd_ms PATHS "${NOORRAY_USD_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
    add_library(usd INTERFACE)
    #! USD's work headers check this macro before they include the version header of oneTBB.
    target_compile_definitions(usd INTERFACE TBB_INTERFACE_VERSION_MAJOR=12)
    target_include_directories(usd SYSTEM INTERFACE "${NOORRAY_USD_INCLUDE_DIR}")
    target_link_libraries(usd INTERFACE "${NOORRAY_USD_LIBRARY}")
endif()
