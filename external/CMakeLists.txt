#! Third-party projects compile without warnings: they are not ours to fix.
set(no_warnings -w)
function(add_external_subdirectory source)
    get_directory_property(options COMPILE_OPTIONS)
    add_compile_options(${no_warnings})
    add_subdirectory(${source} ${ARGN} EXCLUDE_FROM_ALL SYSTEM)
    set_directory_properties(PROPERTIES COMPILE_OPTIONS "${options}")
endfunction()

set(submodules NoorRHI RTXDI stb tinyexr tinygltf tinyobjloader xatlas)
if(NOORRAY_BUILD_APP)
    list(APPEND submodules RmlKit)
endif()
foreach(submodule ${submodules})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${submodule}/.git")
        message(FATAL_ERROR "external/${submodule} is missing. Run: git submodule update --init --recursive")
    endif()
endforeach()

add_subdirectory(NoorRHI EXCLUDE_FROM_ALL)

#! OpenDLSS-RR is a plain Vulkan library: it borrows the device of NoorRHI and records into its command buffers.
if(NOORRAY_OPENDLSS_RR)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/OpenDLSS-RR/.git")
        message(FATAL_ERROR "external/OpenDLSS-RR is missing. Run: git submodule update --init --recursive")
    endif()
    set(SLANGC "${NOORRHI_SLANGC}")
    add_external_subdirectory(OpenDLSS-RR)
endif()

if(NOORRAY_BUILD_APP)
    add_subdirectory(RmlKit EXCLUDE_FROM_ALL)
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

add_library(xatlas STATIC xatlas/source/xatlas/xatlas.cpp)
target_include_directories(xatlas SYSTEM PUBLIC xatlas/source/xatlas)
target_compile_options(xatlas PRIVATE ${no_warnings})

add_library(rtxdi STATIC RTXDI/Source/ReSTIRDI.cpp RTXDI/Source/ReSTIRPT.cpp RTXDI/Source/RtxdiUtils.cpp)
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

#! NRD and FidelityFX stay unmodified upstream: NoorRay compiles their shaders with slangc itself, so their own builds, which need dxc and ShaderMake, are not used.
set(nrd_options NRD_SUPPORTS_VIEWPORT_OFFSET=0 NRD_SUPPORTS_CHECKERBOARD=0 NRD_SUPPORTS_HISTORY_CONFIDENCE=0 NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX=0
    NRD_SUPPORTS_ANTIFIREFLY=1 NRD_SUPPORTS_QUAD_INTRINSICS=0 REBLUR_PERFORMANCE_MODE=0 NRD_NORMAL_ENCODING=4 NRD_ROUGHNESS_ENCODING=1)
#! NRD includes its generated NRDConfig.hlsli from its own Shaders directory, so the build puts it in a Shaders directory beside an include directory.
set(nrd_config "${CMAKE_CURRENT_BINARY_DIR}/NrdConfig/Shaders")
set(nrd_config_text "")
foreach(option IN LISTS nrd_options)
    string(REPLACE "=" " " option "${option}")
    string(APPEND nrd_config_text "#define ${option}\n")
endforeach()
file(CONFIGURE OUTPUT "${nrd_config}/NRDConfig.hlsli" CONTENT "${nrd_config_text}")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/NrdConfig/Include")
set(NOORRAY_NRD_CONFIG_DIR "${nrd_config}" CACHE INTERNAL "")
add_library(nrd STATIC NRD/Source/InstanceImpl.cpp NRD/Source/Reblur.cpp NRD/Source/Reference.cpp NRD/Source/Relax.cpp NRD/Source/Sigma.cpp
    NRD/Source/Timer.cpp NRD/Source/Wrapper.cpp)
target_include_directories(nrd SYSTEM PUBLIC NRD/Include PRIVATE MathLib ShaderMake "${nrd_config}" "${CMAKE_CURRENT_BINARY_DIR}/NrdConfig/Include")
target_compile_definitions(nrd PUBLIC NRD_STATIC_LIBRARY=1 PRIVATE NRD_EMBEDS_SPIRV_SHADERS=0 NRD_EMBEDS_DXIL_SHADERS=0 NRD_EMBEDS_DXBC_SHADERS=0
    SPIRV_SREG_OFFSET=0 SPIRV_BREG_OFFSET=0 SPIRV_UREG_OFFSET=0 SPIRV_TREG_OFFSET=0 NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_features(nrd PRIVATE cxx_std_17)
#! MathLib needs SSSE3; the browser build emulates it with WebAssembly SIMD. __SCE__ selects the x86intrin.h and clock_gettime paths of MathLib and NRD.
if(EMSCRIPTEN)
    target_compile_options(nrd PRIVATE ${no_warnings} -msimd128 -mssse3)
    target_compile_definitions(nrd PRIVATE __SCE__=1)
    target_include_directories(nrd SYSTEM PRIVATE Compat/Emscripten)
else()
    target_compile_options(nrd PRIVATE ${no_warnings} -mssse3)
endif()

set(ffx_sdk FidelityFX-SDK/sdk)
set(ffx_sources ${ffx_sdk}/src/components/fsr3upscaler/ffx_fsr3upscaler.cpp ${ffx_sdk}/src/shared/ffx_assert.cpp ${ffx_sdk}/src/shared/ffx_message.cpp
    ${ffx_sdk}/src/shared/ffx_object_management.cpp)
add_library(fsr3upscaler STATIC ${ffx_sources})
target_include_directories(fsr3upscaler SYSTEM PUBLIC ${ffx_sdk}/include PRIVATE ${ffx_sdk}/src/shared ${ffx_sdk}/src/components/fsr3upscaler)
target_compile_definitions(fsr3upscaler PUBLIC FFX_GCC)
target_compile_features(fsr3upscaler PRIVATE cxx_std_20)
#! The SDK is written for MSVC: FfxPlatform.h supplies the standard headers and the MSVC string functions that it uses without including.
target_compile_options(fsr3upscaler PRIVATE ${no_warnings} -include "${PROJECT_SOURCE_DIR}/src/NoorRay/Render/Upscale/FfxPlatform.h")

#! Blender's permissive sky sources stay unmodified; Compat/BlenderSky replaces their two GPL headers.
add_library(blendersky STATIC BlenderSky/source/sky_single_scattering.cpp BlenderSky/source/sky_multiple_scattering.cpp BlenderSky/source/sky_hosek.cpp)
target_include_directories(blendersky SYSTEM PUBLIC BlenderSky/include Compat/BlenderSky)
target_compile_features(blendersky PRIVATE cxx_std_20)
target_compile_options(blendersky PRIVATE ${no_warnings})
