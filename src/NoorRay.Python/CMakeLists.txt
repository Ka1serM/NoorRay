#! One stable-ABI module loads into every Python from 3.11 on, which covers the Pythons of the supported Blender versions.
set(python_sources
    Module.cpp
    Engine.cpp
    GlInterop.cpp
    Blender/BlenderTextures.cpp
    Blender/GridSet.cpp
    Blender/MaterialReader.cpp
    Blender/MeshReader.cpp
    Blender/NodeTreeReader.cpp
    Blender/Parallel.cpp
    Blender/Reader.cpp
    Blender/SceneSync.cpp
    Blender/SkyTexture.cpp
    Blender/StrandReader.cpp
    Blender/TextureTable.cpp
    Blender/VolumeReader.cpp)
if(PLATFORM_WINDOWS)
    #! llvm-mingw has no Windows Python, so the headers and python3.lib come from the official package; python3.dll ships with every Blender since 4.2.
    include(FetchContent)
    FetchContent_Declare(python_windows
                         URL https://api.nuget.org/v3-flatcontainer/python/3.11.9/python.3.11.9.nupkg
                         URL_HASH SHA256=9283876d58c017e0e846f95b490da3bca0fc0a6ee1134b2870677cfb7eec3c67
                         DOWNLOAD_NAME python.3.11.9.zip
                         DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(python_windows)
    add_library(NoorRay.Python MODULE ${python_sources})
    target_include_directories(NoorRay.Python SYSTEM PRIVATE "${python_windows_SOURCE_DIR}/tools/include")
    #! pyconfig.h sets MS_WIN64 for MSVC only.
    target_compile_definitions(NoorRay.Python PRIVATE Py_LIMITED_API=0x030B0000 MS_WIN64)
    target_link_libraries(NoorRay.Python PRIVATE "${python_windows_SOURCE_DIR}/tools/libs/python3.lib")
    set_target_properties(NoorRay.Python PROPERTIES PREFIX "" SUFFIX ".pyd")
else()
    find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.SABIModule)
    Python_add_library(NoorRay.Python MODULE USE_SABI 3.11 WITH_SOABI ${python_sources})
endif()
noorray_code_options(NoorRay.Python)
target_compile_features(NoorRay.Python PRIVATE cxx_std_23)
target_link_libraries(NoorRay.Python PRIVATE NoorRay::NoorRay blendersky)
noorray_link_denoiser_plugins(NoorRay.Python)
#! Blender does not ship libc++, so the module carries its own C++ runtime and keeps those symbols to itself.
if(PLATFORM_LINUX)
    target_link_options(NoorRay.Python PRIVATE -nostdlib++ -static-libgcc LINKER:--exclude-libs,ALL
                        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/Exports.map")
    target_link_libraries(NoorRay.Python PRIVATE -l:libc++.a -l:libc++abi.a)
    set_property(TARGET NoorRay.Python APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Exports.map")
endif()
set_target_properties(NoorRay.Python PROPERTIES OUTPUT_NAME noorray CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
