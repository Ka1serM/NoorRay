# The gpu library is self-contained: Vulkan, Vulkan-HPP and VMA are its only
# dependencies. It knows nothing about NoorRay, shader compilers or assets.
add_library(noorray_gpu STATIC
    src/device.cpp
    src/graphics.cpp
    src/swapchain.cpp
    src/image.cpp
    src/internal.hpp)

set_target_properties(noorray_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(noorray_gpu PUBLIC cxx_std_23)

# Only the public headers are visible to consumers; src/ holds internal.hpp,
# which pulls in Vulkan-HPP and VMA and must not leak into dependents.
target_include_directories(noorray_gpu
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_compile_definitions(noorray_gpu PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
target_link_libraries(noorray_gpu PUBLIC Vulkan::Vulkan PRIVATE NoorRayVma)
