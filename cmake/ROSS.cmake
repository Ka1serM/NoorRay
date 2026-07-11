# NoorRay-owned integration for the unmodified ROSS submodule.
set(ROSS_ROOT "${PROJECT_SOURCE_DIR}/external/ROSS")

set(ROSS_BUILD_GPU_SUPPORT ON)
set(ROSS_USE_UNITY_BUILD OFF)

set(FMT_TEST OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CPPTRACE_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CPPTRACE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LIB_MORTON_TESTING OFF CACHE BOOL "" FORCE)

add_subdirectory("${ROSS_ROOT}/ext/fmt" "${CMAKE_BINARY_DIR}/ross/fmt" EXCLUDE_FROM_ALL)
add_subdirectory("${ROSS_ROOT}/ext/spdlog" "${CMAKE_BINARY_DIR}/ross/spdlog" EXCLUDE_FROM_ALL)
add_subdirectory("${ROSS_ROOT}/ext/nlohmann-json" "${CMAKE_BINARY_DIR}/ross/nlohmann-json" EXCLUDE_FROM_ALL)
add_subdirectory("${ROSS_ROOT}/ext/cli11" "${CMAKE_BINARY_DIR}/ross/cli11" EXCLUDE_FROM_ALL)
add_subdirectory("${ROSS_ROOT}/ext/cpptrace" "${CMAKE_BINARY_DIR}/ross/cpptrace" EXCLUDE_FROM_ALL)
add_subdirectory("${ROSS_ROOT}/ext/magic_enum" "${CMAKE_BINARY_DIR}/ross/magic_enum" EXCLUDE_FROM_ALL)
add_subdirectory("${ROSS_ROOT}/ext/tabulate" "${CMAKE_BINARY_DIR}/ross/tabulate" EXCLUDE_FROM_ALL)
set(_NR_BUILD_TESTING "${BUILD_TESTING}")
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory("${ROSS_ROOT}/ext/libmorton" "${CMAKE_BINARY_DIR}/ross/libmorton" EXCLUDE_FROM_ALL)
set(BUILD_TESTING "${_NR_BUILD_TESTING}" CACHE BOOL "" FORCE)
unset(_NR_BUILD_TESTING)
add_subdirectory("${ROSS_ROOT}/ext/ordered_map" "${CMAKE_BINARY_DIR}/ross/ordered_map" EXCLUDE_FROM_ALL)
add_subdirectory("${ROSS_ROOT}/ext/gilbert" "${CMAKE_BINARY_DIR}/ross/gilbert" EXCLUDE_FROM_ALL)
add_subdirectory("${ROSS_ROOT}/ext/lodepng" "${CMAKE_BINARY_DIR}/ross/lodepng" EXCLUDE_FROM_ALL)

foreach(_nr_pic_target
        fmt
        spdlog
        nlohmann_json
        CLI11
        cpptrace-lib
        magic_enum
        tabulate
        morton
        gilbert
        lodepng)
    if(TARGET ${_nr_pic_target})
        set_target_properties(${_nr_pic_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
endforeach()

add_library(hash-library STATIC
    "${ROSS_ROOT}/ext/hash-library/crc32.cpp"
    "${ROSS_ROOT}/ext/hash-library/digest.cpp"
    "${ROSS_ROOT}/ext/hash-library/keccak.cpp"
    "${ROSS_ROOT}/ext/hash-library/md5.cpp"
    "${ROSS_ROOT}/ext/hash-library/sha1.cpp"
    "${ROSS_ROOT}/ext/hash-library/sha256.cpp"
    "${ROSS_ROOT}/ext/hash-library/sha3.cpp")
set_target_properties(hash-library PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(hash-library PUBLIC "${ROSS_ROOT}/ext/hash-library")

add_subdirectory("${ROSS_ROOT}/openlensfileio/ext/text_encoding_detect"
                 "${CMAKE_BINARY_DIR}/ross/text_encoding_detect" EXCLUDE_FROM_ALL)
add_subdirectory("${ROSS_ROOT}/openlensfileio/src/openlensfileio/main"
                 "${CMAKE_BINARY_DIR}/ross/openlensfileio" EXCLUDE_FROM_ALL)
foreach(_nr_pic_target text_encoding_detect openlensfileio)
    if(TARGET ${_nr_pic_target})
        set_target_properties(${_nr_pic_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
endforeach()

set(OPENEXR_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(OPENEXR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(OPENEXR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(OPENEXR_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${ROSS_ROOT}/pbrt-v4/src/ext/openexr"
                 "${CMAKE_BINARY_DIR}/ross/openexr" EXCLUDE_FROM_ALL)
add_subdirectory("${ROSS_ROOT}/libross/main" "${CMAKE_BINARY_DIR}/ross/libross" EXCLUDE_FROM_ALL)

target_compile_definitions(libross PUBLIC ROSS_BUILD_GPU_SUPPORT=1)
target_link_libraries(libross PUBLIC CUDA::cudart)
set_target_properties(libross PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${NR_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(libross PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(libross SYSTEM PUBLIC
    "${CUDAToolkit_INCLUDE_DIRS}"
    "${CUDAToolkit_TARGET_DIR}/include/cccl")
