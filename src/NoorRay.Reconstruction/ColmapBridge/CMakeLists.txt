cmake_minimum_required(VERSION 3.29)
#! A project of its own: COLMAP is a libstdc++ library with exceptions, which NoorRay's libc++ build without exceptions cannot link.
project(NoorRayColmapBridge LANGUAGES C CXX)

#! COLMAP's package enables CUDA, and nvcc needs the GCC that built COLMAP as its host compiler.
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "")
find_package(COLMAP REQUIRED)

add_library(NoorRayColmap SHARED ColmapBridge.cpp)
target_compile_features(NoorRayColmap PRIVATE cxx_std_17)
target_link_libraries(NoorRayColmap PRIVATE colmap::colmap)
#! Only the C functions leave the library, so COLMAP and its libstdc++ symbols cannot meet those of the app.
set_target_properties(NoorRayColmap PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
target_link_options(NoorRayColmap PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map" -Wl,-Bsymbolic -Wl,--no-undefined)
install(TARGETS NoorRayColmap LIBRARY DESTINATION .)
