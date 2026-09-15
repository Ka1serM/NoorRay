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

add_noorray_test(kolb_lens_test
    "unit/optics/KolbLensTest.cpp;${CMAKE_SOURCE_DIR}/src/libnoorray/Optics/KolbLens.cpp" optics)
target_compile_definitions(kolb_lens_test PRIVATE
    OPTICS_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets/optics")
target_link_libraries(kolb_lens_test PRIVATE glm)

add_noorray_test(pbrt_parser_test
    "unit/scene/PbrtParserTest.cpp;${CMAKE_SOURCE_DIR}/src/libnoorray/Scene/Import/PbrtParser.cpp" pbrt)


# The NoorRHI API's own tests live with the library in external/NoorRHI/tests and run from
# its standalone build. What stays here is NoorRay-specific: the composite BSDF
# shader (which includes src/libnoorray/Shaders) and the adoption test, which
# checks NoorRHI links cleanly inside NoorRay's archive closure.
set(NR_NOORRHI_TEST_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/noorrhi")
set(NR_NOORRHI_ADD_SHADER "${NR_NOORRHI_TEST_SHADER_DIR}/add.spv")
set(NR_NOORRHI_COMPOSITE_BSDF_SHADER "${NR_NOORRHI_TEST_SHADER_DIR}/composite_bsdf.spv")
file(GLOB NR_NOORRHI_RAYTRACER_SHADER_DEPENDENCIES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/libnoorray/Shaders/Raytracer/*.slang")

set(NR_NOORRHI_SHADER_FLAGS
    -emit-spirv-directly
    -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081)

add_custom_command(
    OUTPUT "${NR_NOORRHI_ADD_SHADER}" "${NR_NOORRHI_COMPOSITE_BSDF_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${NR_NOORRHI_TEST_SHADER_DIR}"
    COMMAND "${NR_SLANGC_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/external/NoorRHI/tests/shaders/add.slang"
        -entry main -stage compute -target spirv
        -profile spirv_1_5 ${NR_NOORRHI_SHADER_FLAGS}
        -o "${NR_NOORRHI_ADD_SHADER}"
    COMMAND "${NR_SLANGC_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/noorrhi/composite_bsdf.slang"
        -entry main -stage compute -target spirv
        -profile spirv_1_5 ${NR_NOORRHI_SHADER_FLAGS}
        -o "${NR_NOORRHI_COMPOSITE_BSDF_SHADER}"
    DEPENDS "${CMAKE_SOURCE_DIR}/external/NoorRHI/tests/shaders/add.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/noorrhi/composite_bsdf.slang"
        ${NR_NOORRHI_RAYTRACER_SHADER_DEPENDENCIES}
    COMMENT "Compiling NoorRay NoorRHI test shaders" VERBATIM)
add_custom_target(noorray_noorrhi_test_shader DEPENDS
    "${NR_NOORRHI_ADD_SHADER}" "${NR_NOORRHI_COMPOSITE_BSDF_SHADER}")

add_noorray_test(composite_bsdf_shader_test unit/materials/CompositeBsdfShaderTest.cpp materials)
add_dependencies(composite_bsdf_shader_test noorray_noorrhi_test_shader)
target_compile_definitions(composite_bsdf_shader_test PRIVATE
    NOORRHI_COMPOSITE_BSDF_SHADER="${NR_NOORRHI_COMPOSITE_BSDF_SHADER}")
target_link_libraries(composite_bsdf_shader_test PRIVATE NoorRHI::NoorRHI)

add_noorray_test(noorrhi_adoption_test unit/noorrhi/NoorRhiAdoptionTest.cpp NoorRHI)
add_dependencies(noorrhi_adoption_test noorray_noorrhi_test_shader)
target_compile_definitions(noorrhi_adoption_test PRIVATE NOORRHI_TEST_SHADER="${NR_NOORRHI_ADD_SHADER}")
target_link_libraries(noorrhi_adoption_test PRIVATE
    "$<LINK_GROUP:RESCAN,libnoorray,NoorRayIO,NoorRHI>")

add_noorray_test(vertex_color_test unit/scene/VertexColorTest.cpp vertex-color)
target_link_libraries(vertex_color_test PRIVATE glm)

add_noorray_test(render_settings_test unit/scene/RenderSettingsTest.cpp scene)
target_link_libraries(render_settings_test PRIVATE glm)

add_noorray_test(texture_storage_test
    unit/scene/TextureStorageTest.cpp texture)
target_link_libraries(texture_storage_test PRIVATE
    "$<LINK_GROUP:RESCAN,libnoorray,NoorRayIO>")

add_noorray_test(imgui_math_input_test
    "unit/ui/ImGuiMathInputTest.cpp;${CMAKE_SOURCE_DIR}/src/app/UI/MathInput.cpp" ui)
target_include_directories(imgui_math_input_test PRIVATE "${CMAKE_SOURCE_DIR}/src/app")
target_link_libraries(imgui_math_input_test PRIVATE imgui)

add_noorray_test(viewport_billboard_picking_test
    unit/ui/ViewportBillboardPickingTest.cpp ui)
target_link_libraries(viewport_billboard_picking_test PRIVATE glm NoorRHI::NoorRHI)

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
${CMAKE_SOURCE_DIR}/src/app/UI/MaterialXNodes/MaterialXMathGraphNode.cpp;\
${CMAKE_SOURCE_DIR}/src/app/UI/MaterialXNodes/MaterialXSellmeierGraphNode.cpp" ui)
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

add_noorray_test(scene_importer_cache_test unit/scene/SceneImporterCacheTest.cpp scene)
target_compile_definitions(scene_importer_cache_test PRIVATE
    TEST_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")
target_link_libraries(scene_importer_cache_test PRIVATE
    "$<LINK_GROUP:RESCAN,libnoorray,NoorRayIO>")

if(NR_ENABLE_USD)
    add_noorray_test(scene_usd_test unit/scene/SceneUsdTest.cpp scene)
    target_link_libraries(scene_usd_test PRIVATE
        "$<LINK_GROUP:RESCAN,libnoorray,NoorRayIO>"
        NoorRayUsdStandalone)
endif()

add_noorray_test(scene_pbrt_test unit/scene/ScenePbrtTest.cpp scene)
target_link_libraries(scene_pbrt_test PRIVATE
    "$<LINK_GROUP:RESCAN,libnoorray,NoorRayIO>")

function(add_noorray_render_test target source)
    if(NOT TARGET NoorRay)
        return()
    endif()
    add_noorray_test(${target} ${source} e2e)
    target_compile_definitions(${target} PRIVATE
        TEST_SCENE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
        NOORRAY_EXECUTABLE="$<TARGET_FILE:NoorRay>"
    )
    target_link_libraries(${target} PRIVATE NoorRayIO)
    add_dependencies(${target} NoorRay)
endfunction()

add_noorray_test(editor_edit_render_test e2e/EditorEditRenderTest.cpp e2e)
target_link_libraries(editor_edit_render_test PRIVATE
    "$<LINK_GROUP:RESCAN,libnoorray,NoorRayIO,NoorRHI,NoorRayMaterialX>")
add_noorray_render_test(render_smoke_test e2e/RenderSmokeTest.cpp)
add_noorray_render_test(white_furnace_test e2e/WhiteFurnaceTest.cpp)
add_noorray_render_test(convergence_test e2e/ConvergenceTest.cpp)

add_noorray_render_test(materialx_render_test e2e/MaterialXRenderTest.cpp)
add_noorray_render_test(materialx_fidelity_test e2e/MaterialXFidelityTest.cpp)
set(MATERIALXVIEW_EXECUTABLE
    "${CMAKE_BINARY_DIR}/../materialxview/bin/MaterialXView"
    CACHE FILEPATH "MaterialXView executable used by the C++ fidelity test")
if(TARGET materialx_fidelity_test)
    target_compile_definitions(materialx_fidelity_test PRIVATE
        TEST_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
        MATERIALXVIEW_EXECUTABLE="${MATERIALXVIEW_EXECUTABLE}"
        MATERIALX_RESOURCE_DIR="${CMAKE_SOURCE_DIR}/external/materialx/resources"
        MATERIALX_SOURCE_DIR="${CMAKE_SOURCE_DIR}/external/materialx")
endif()
