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
    "unit/cuda/TaggedObjectTest.cpp;${CMAKE_SOURCE_DIR}/src/libnoorray/Backend/CUDA/rstd/Memory.cpp" tagged)
target_compile_definitions(tagged_object_test PRIVATE NR_BACKEND_CUDA=1)
target_include_directories(tagged_object_test PRIVATE "${OPTIX_INCLUDE_DIR}")
target_link_libraries(tagged_object_test PRIVATE CUDA::cudart)

add_noorray_test(pbrt_parser_test
    "unit/scene/PbrtParserTest.cpp;${CMAKE_SOURCE_DIR}/src/libnoorray/Scene/Import/PbrtParser.cpp" pbrt)

add_noorray_test(ply_mesh_loader_test
    "unit/scene/PlyMeshLoaderTest.cpp;${CMAKE_SOURCE_DIR}/src/libnoorray/Geometry/Mesh/Assets/PlyMeshLoader.cpp" ply)
target_include_directories(ply_mesh_loader_test PRIVATE
    "${OPTIX_INCLUDE_DIR}")
target_compile_definitions(ply_mesh_loader_test PRIVATE
    NR_BACKEND_CUDA=1
    PLY_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")
target_link_libraries(ply_mesh_loader_test PRIVATE
    tinyply glm Vulkan::Vulkan CUDA::cudart)

add_noorray_test(vertex_color_test unit/scene/VertexColorTest.cpp vertex-color)
target_link_libraries(vertex_color_test PRIVATE glm)

add_noorray_test(render_settings_test unit/scene/RenderSettingsTest.cpp scene)
target_include_directories(render_settings_test PRIVATE
    "${OPTIX_INCLUDE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}")
target_compile_definitions(render_settings_test PRIVATE NR_BACKEND_CUDA=1)
target_link_libraries(render_settings_test PRIVATE glm CUDA::cudart)

add_noorray_test(texture_storage_test
    unit/scene/TextureStorageTest.cpp texture)
target_link_libraries(texture_storage_test PRIVATE
    "$<LINK_GROUP:RESCAN,libnoorray,NoorRayCuda,NoorRayIO,libross>"
    CUDA::cudart_static)

add_noorray_test(active_bsdf_test unit/shading/ActiveBsdfTest.cpp bsdf)
target_link_libraries(active_bsdf_test PRIVATE glm CUDA::cudart)

add_noorray_test(diffuse_lobe_test unit/shading/DiffuseLobeTest.cpp bsdf)
target_link_libraries(diffuse_lobe_test PRIVATE glm CUDA::cudart)

add_noorray_test(composite_bsdf_test unit/shading/CompositeBsdfTest.cpp bsdf)
target_link_libraries(composite_bsdf_test PRIVATE glm CUDA::cudart)

add_noorray_test(light_sampling_test unit/shading/LightSamplingTest.cpp light)
target_link_libraries(light_sampling_test PRIVATE glm CUDA::cudart)

add_noorray_test(shadow_terminator_test unit/raytracing/ShadowTerminatorTest.cpp raytracing)
target_link_libraries(shadow_terminator_test PRIVATE glm)

add_noorray_test(ray_test unit/raytracing/RayTest.cpp raytracing)
target_link_libraries(ray_test PRIVATE glm)

add_noorray_test(imgui_math_input_test
    "unit/ui/ImGuiMathInputTest.cpp;${CMAKE_SOURCE_DIR}/src/app/UI/MathInput.cpp" ui)
target_include_directories(imgui_math_input_test PRIVATE "${CMAKE_SOURCE_DIR}/src/app")
target_link_libraries(imgui_math_input_test PRIVATE imgui)

add_noorray_test(materialx_node_catalog_test
    "unit/ui/MaterialXNodeCatalogTest.cpp;${CMAKE_SOURCE_DIR}/src/app/UI/MaterialXNodeCatalog.cpp" ui)
target_include_directories(materialx_node_catalog_test PRIVATE "${CMAKE_SOURCE_DIR}/src/app")
target_compile_definitions(materialx_node_catalog_test PRIVATE
    NR_MATERIALX_STDLIB_DIR="${NR_MATERIALX_STDLIB_DIR}")
target_link_libraries(materialx_node_catalog_test PRIVATE NoorRayMaterialX)

add_noorray_test(svm_materialx_compiler_test
    "unit/svm/SvmMaterialXCompilerTest.cpp;${CMAKE_SOURCE_DIR}/src/libnoorray/Materials/SVM/SvmMaterialXCompiler.cpp" svm)
target_compile_definitions(svm_materialx_compiler_test PRIVATE
    NR_MATERIALX_STDLIB_DIR="${NR_MATERIALX_STDLIB_DIR}")
target_link_libraries(svm_materialx_compiler_test PRIVATE
    MaterialXCore MaterialXFormat MaterialXGenShader NoorRayMaterialX glm)

# Every node class file is listed: they register themselves at start-up and are
# not referenced from anywhere, so the test only covers what it links.
add_noorray_test(materialx_graph_node_registry_test
    "unit/ui/MaterialXGraphNodeRegistryTest.cpp;\
${CMAKE_SOURCE_DIR}/src/app/UI/MaterialXNodes/MaterialXGraphNode.cpp;\
${CMAKE_SOURCE_DIR}/src/app/UI/MaterialXNodes/MaterialXGraphNodeRegistry.cpp;\
${CMAKE_SOURCE_DIR}/src/app/UI/MaterialXNodeCatalog.cpp;\
${CMAKE_SOURCE_DIR}/src/app/UI/MaterialXNodes/MaterialXImageGraphNode.cpp;\
${CMAKE_SOURCE_DIR}/src/app/UI/MaterialXNodes/MaterialXConstantGraphNode.cpp;\
${CMAKE_SOURCE_DIR}/src/app/UI/MaterialXNodes/MaterialXShaderGraphNode.cpp;\
${CMAKE_SOURCE_DIR}/src/app/UI/MaterialXNodes/MaterialXMathGraphNode.cpp" ui)
target_include_directories(materialx_graph_node_registry_test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/app")
target_compile_definitions(materialx_graph_node_registry_test PRIVATE
    NR_MATERIALX_STDLIB_DIR="${NR_MATERIALX_STDLIB_DIR}")
target_link_libraries(materialx_graph_node_registry_test PRIVATE
    imgui imnodeflow MaterialXCore NoorRayMaterialX)

add_noorray_test(materialx_graph_layout_test
    "unit/ui/MaterialXGraphLayoutTest.cpp;${CMAKE_SOURCE_DIR}/src/app/UI/MaterialXNodes/MaterialXGraphLayout.cpp" ui)
target_include_directories(materialx_graph_layout_test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/app")

add_noorray_test(scene_lifetime_test unit/scene/SceneLifetimeTest.cpp scene)
target_link_libraries(scene_lifetime_test PRIVATE
    "$<LINK_GROUP:RESCAN,libnoorray,NoorRayCuda,NoorRayIO,libross>"
    CUDA::cudart_static NoorRayUsdStandalone)

add_noorray_test(resource_registry_test
    unit/scene/ResourceRegistryTest.cpp scene)

add_noorray_test(scene_importer_cache_test unit/scene/SceneImporterCacheTest.cpp scene)
target_compile_definitions(scene_importer_cache_test PRIVATE
    TEST_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")
target_link_libraries(scene_importer_cache_test PRIVATE
    "$<LINK_GROUP:RESCAN,libnoorray,NoorRayCuda,NoorRayIO,libross>"
    CUDA::cudart_static NoorRayUsdStandalone)

add_noorray_test(scene_usd_test unit/scene/SceneUsdTest.cpp scene)
target_link_libraries(scene_usd_test PRIVATE
    "$<LINK_GROUP:RESCAN,libnoorray,NoorRayCuda,NoorRayIO,libross>"
    CUDA::cudart_static NoorRayUsdStandalone)

add_noorray_test(scene_pbrt_test unit/scene/ScenePbrtTest.cpp scene)
target_link_libraries(scene_pbrt_test PRIVATE
    "$<LINK_GROUP:RESCAN,libnoorray,NoorRayCuda,NoorRayIO,libross>"
    CUDA::cudart_static NoorRayUsdStandalone)

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

add_noorray_render_test(materialx_fidelity_test e2e/MaterialXFidelityTest.cpp)
set(MATERIALXVIEW_EXECUTABLE
    "${CMAKE_BINARY_DIR}/../materialxview/bin/MaterialXView"
    CACHE FILEPATH "MaterialXView executable used by the C++ fidelity test")
target_compile_definitions(materialx_fidelity_test PRIVATE
    TEST_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
    MATERIALXVIEW_EXECUTABLE="${MATERIALXVIEW_EXECUTABLE}"
    MATERIALX_RESOURCE_DIR="${CMAKE_SOURCE_DIR}/external/materialx/resources"
    MATERIALX_SOURCE_DIR="${CMAKE_SOURCE_DIR}/external/materialx")
