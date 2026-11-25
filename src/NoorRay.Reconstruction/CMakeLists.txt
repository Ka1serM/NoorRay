file(GLOB_RECURSE reconstruction_sources CONFIGURE_DEPENDS *.cpp)
#! ColmapBridge is a project of its own with libstdc++; see its CMakeLists.txt.
list(FILTER reconstruction_sources EXCLUDE REGEX "/ColmapBridge/")
add_library(NoorRay.Reconstruction STATIC ${reconstruction_sources})
noorray_code_options(NoorRay.Reconstruction)
target_compile_features(NoorRay.Reconstruction PRIVATE cxx_std_23)
target_compile_definitions(NoorRay.Reconstruction PRIVATE PRODUCT_NAME="NoorRay" NOORRAY_DATASETS="${NOORRAY_DATASETS}")
target_include_directories(NoorRay.Reconstruction PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(NoorRay.Reconstruction PRIVATE NoorRay::NoorRay DiffRenderer::DiffRenderer RmlKit::RmlKit SDL3::SDL3-static)

file(GLOB_RECURSE reconstruction_ui CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/UI/*.rml" "${CMAKE_CURRENT_SOURCE_DIR}/UI/*.rcss" "${CMAKE_CURRENT_SOURCE_DIR}/UI/*.svg")
set_property(GLOBAL APPEND PROPERTY NOORRAY_PLUGINS NoorRay.Reconstruction)
set_property(GLOBAL APPEND PROPERTY NOORRAY_PLUGIN_RESOURCES ROOT "${CMAKE_CURRENT_SOURCE_DIR}" ${reconstruction_ui})

#! COLMAP 4 links only with the GCC that built it, so its bridge is an external project with that compiler.
set(NOORRAY_COLMAP_CXX_COMPILER "g++-15" CACHE STRING "The C++ compiler that built the COLMAP libraries; the COLMAP bridge builds with it.")
set(NOORRAY_COLMAP_C_COMPILER "gcc-15" CACHE STRING "The C compiler that matches NOORRAY_COLMAP_CXX_COMPILER.")
find_program(NOORRAY_NVCC nvcc PATHS /usr/local/cuda/bin)
set(colmap_bridge_arguments -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=${NOORRAY_COLMAP_C_COMPILER} -DCMAKE_CXX_COMPILER=${NOORRAY_COLMAP_CXX_COMPILER}
    -DCMAKE_PREFIX_PATH=${NOORRAY_COLMAP_PREFIX} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -Wno-dev)
if(NOORRAY_NVCC)
    list(APPEND colmap_bridge_arguments -DCMAKE_CUDA_COMPILER=${NOORRAY_NVCC})
endif()
include(ExternalProject)
set(colmap_bridge_library "${CMAKE_CURRENT_BINARY_DIR}/ColmapBridge/install/libNoorRayColmap.so")
ExternalProject_Add(NoorRay.ColmapBridge
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ColmapBridge"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/ColmapBridge/build"
    INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/ColmapBridge/install"
    CMAKE_ARGS ${colmap_bridge_arguments}
    BUILD_ALWAYS ON
    BUILD_BYPRODUCTS "${colmap_bridge_library}")
add_dependencies(NoorRay.Reconstruction NoorRay.ColmapBridge)
set_property(GLOBAL APPEND PROPERTY NOORRAY_PLUGIN_LIBRARIES "${colmap_bridge_library}")
