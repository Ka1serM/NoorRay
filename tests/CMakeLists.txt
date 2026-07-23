add_subdirectory("${CMAKE_SOURCE_DIR}/external/Catch2" "${CMAKE_BINARY_DIR}/external/Catch2")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/Catch2/extras")
include(Catch)

function(add_noorray_test target source prefix)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/src/libnoorray"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(${target} PRIVATE Catch2::Catch2WithMain)
    set_target_properties(${target} PROPERTIES FOLDER "Tests/${prefix}")
    catch_discover_tests(${target}
        TEST_PREFIX "${prefix}:"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        PROPERTIES LABELS "${prefix}"
    )
endfunction()

add_noorray_test(sampler_test unit/sampling/SamplerTest.cpp sampler)
target_link_libraries(sampler_test PRIVATE glm)

add_noorray_test(cuda_unique_handle_test unit/cuda/CudaUniqueHandleTest.cpp cuda)
add_noorray_test(tagged_object_test
    "unit/cuda/TaggedObjectTest.cpp;${CMAKE_SOURCE_DIR}/src/libnoorray/CUDA/rstd/Memory.cpp" tagged)
target_compile_definitions(tagged_object_test PRIVATE NR_BACKEND_CUDA=1)
target_include_directories(tagged_object_test PRIVATE "${OPTIX_INCLUDE_DIR}")
target_link_libraries(tagged_object_test PRIVATE CUDA::cudart)

add_noorray_test(pbrt_parser_test
    "unit/scene/PbrtParserTest.cpp;${CMAKE_SOURCE_DIR}/src/libnoorray/Scene/PbrtParser.cpp" pbrt)

add_noorray_test(ply_mesh_loader_test
    "unit/scene/PlyMeshLoaderTest.cpp;${CMAKE_SOURCE_DIR}/src/libnoorray/Mesh/Assets/PlyMeshLoader.cpp" ply)
target_include_directories(ply_mesh_loader_test PRIVATE
    "${OPTIX_INCLUDE_DIR}")
target_compile_definitions(ply_mesh_loader_test PRIVATE
    NR_BACKEND_CUDA=1
    PLY_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")
target_link_libraries(ply_mesh_loader_test PRIVATE
    tinyply glm Vulkan::Vulkan CUDA::cudart)

add_noorray_test(opaque_bsdf_test unit/shading/OpaqueBsdfTest.cpp bsdf)
target_link_libraries(opaque_bsdf_test PRIVATE glm CUDA::cudart)

add_noorray_test(ggx_bsdf_test unit/shading/GgxBsdfTest.cpp bsdf)
target_link_libraries(ggx_bsdf_test PRIVATE glm CUDA::cudart)

add_noorray_test(dielectric_bsdf_test unit/shading/DielectricBsdfTest.cpp bsdf)
target_link_libraries(dielectric_bsdf_test PRIVATE glm CUDA::cudart)

add_noorray_test(light_sampling_test unit/shading/LightSamplingTest.cpp light)
target_link_libraries(light_sampling_test PRIVATE glm CUDA::cudart)

add_noorray_test(shadow_terminator_test unit/raytracing/ShadowTerminatorTest.cpp raytracing)
target_link_libraries(shadow_terminator_test PRIVATE glm)

add_noorray_test(ray_test unit/raytracing/RayTest.cpp raytracing)
target_link_libraries(ray_test PRIVATE glm)

add_noorray_test(scene_lifetime_test unit/scene/SceneLifetimeTest.cpp scene)
target_link_libraries(scene_lifetime_test PRIVATE
    "$<LINK_GROUP:RESCAN,libnoorray,NoorRayCuda,NoorRayIO,libross>"
    CUDA::cudart_static)

function(add_noorray_render_test target source)
    add_noorray_test(${target} ${source} e2e)
    target_compile_definitions(${target} PRIVATE
        TEST_SCENE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
        NOORRAY_EXECUTABLE="$<TARGET_FILE:NoorRay>"
    )
    target_link_libraries(${target} PRIVATE NoorRayIO)
    add_dependencies(${target} NoorRay)
endfunction()

add_noorray_render_test(render_smoke_test e2e/RenderSmokeTest.cpp)
add_noorray_render_test(white_furnace_test e2e/WhiteFurnaceTest.cpp)
