#include "NoorRaySession.h"
#include "Scene/Scene.h"
#include "Scene/SceneImporter.h"
#include "Scene/SceneObject.h"
#include "Shading/Material.h"

#include <glm/vec3.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE(
    "re-importing the same OBJ file shares its MeshAsset instead of re-uploading it",
    "[scene][import][dedup]")
{
    noorray::NoorRaySession session;
    Scene& scene = session.scene;

    SceneImporter::ImportObjScene(scene, TEST_ASSET_DIR "/dedup_triangle.obj", nullptr);
    const size_t meshCountAfterFirst = scene.getMeshAssets().size();
    const size_t objectCountAfterFirst = scene.getSceneObjects().size();
    REQUIRE(meshCountAfterFirst > 0);

    SceneImporter::ImportObjScene(scene, TEST_ASSET_DIR "/dedup_triangle.obj", nullptr);
    const size_t meshCountAfterSecond = scene.getMeshAssets().size();
    const size_t objectCountAfterSecond = scene.getSceneObjects().size();

    // The second import must not create any new mesh assets -- it should
    // clone the hierarchy the first import already built, sharing the same
    // MeshAssetRef rather than re-parsing the file and re-uploading it.
    CHECK(meshCountAfterSecond == meshCountAfterFirst);
    // It does still create a new (parallel) set of SceneObjects/MeshInstances
    // so the two imports can be transformed/removed independently.
    CHECK(objectCountAfterSecond > objectCountAfterFirst);
}

TEST_CASE(
    "a material override on OBJ import skips the file-import cache",
    "[scene][import][dedup]")
{
    noorray::NoorRaySession session;
    Scene& scene = session.scene;

    SceneImporter::ImportObjScene(scene, TEST_ASSET_DIR "/dedup_triangle.obj", nullptr);
    const size_t meshCountAfterFirst = scene.getMeshAssets().size();

    MaterialAuthoring materialOverride{};
    materialOverride.albedo = glm::vec3(1.0f, 0.0f, 0.0f);
    SceneImporter::ImportObjScene(
        scene, TEST_ASSET_DIR "/dedup_triangle.obj", &materialOverride);
    const size_t meshCountAfterSecond = scene.getMeshAssets().size();

    // An override must never be silently dropped by reusing a cached
    // hierarchy that was built under a different (or no) override.
    CHECK(meshCountAfterSecond > meshCountAfterFirst);
}

TEST_CASE(
    "OBJ preparation preserves geometry and shares one source material across shapes",
    "[scene][import][prepare]")
{
    noorray::NoorRaySession session;
    Scene& scene = session.scene;

    SceneImporter::ImportObjScene(
        scene, TEST_ASSET_DIR "/shared_default_material_shapes.obj", nullptr);

    REQUIRE(scene.getMeshAssets().size() == 2);
    REQUIRE(scene.getMaterials().size() == 1);
    const MeshAsset& first = scene.getMeshAssets()[0];
    const MeshAsset& second = scene.getMeshAssets()[1];
    CHECK(first.getVertices().size() == 3);
    CHECK(first.getIndices().size() == 3);
    CHECK(first.getFaces().size() == 1);
    CHECK(second.getVertices().size() == 3);
    CHECK(second.getIndices().size() == 3);
    CHECK(second.getFaces().size() == 1);
    CHECK(first.getMaterialHandle(0) == second.getMaterialHandle(0));
}

TEST_CASE("MeshAsset adopts a prepared managed payload without copying it",
    "[scene][mesh][prepare]")
{
    noorray::NoorRaySession session;
    Scene& scene = session.scene;

    MeshGeometry geometry;
    geometry.vertices.push_back(Vertex{
        .position = {-0.5f, 0.0f, 0.0f},
        .normal = {0.0f, 0.0f, 1.0f},
        .tangent = {1.0f, 0.0f, 0.0f},
        .tangentSign = 1.0f,
        .uv = {0.0f, 0.0f},
    });
    geometry.vertices.push_back(Vertex{
        .position = {0.5f, 0.0f, 0.0f},
        .normal = {0.0f, 0.0f, 1.0f},
        .tangent = {1.0f, 0.0f, 0.0f},
        .tangentSign = 1.0f,
        .uv = {1.0f, 0.0f},
    });
    geometry.vertices.push_back(Vertex{
        .position = {0.0f, 1.0f, 0.0f},
        .normal = {0.0f, 0.0f, 1.0f},
        .tangent = {1.0f, 0.0f, 0.0f},
        .tangentSign = 1.0f,
        .uv = {0.5f, 1.0f},
    });
    geometry.indices = {0, 1, 2};
    geometry.faces = {Face{0}};
    const Vertex* const vertices = geometry.vertices.data();
    const uint32_t* const indices = geometry.indices.data();
    const Face* const faces = geometry.faces.data();

    const MaterialRef material = scene.add(Material{});
    std::vector<MaterialRef> materials{material};
    const MeshAssetRef asset = scene.add(MeshAsset(scene, "Prepared",
        std::move(geometry), std::move(materials)));

    REQUIRE(asset.isValid());
    CHECK(asset.get()->getVertices().data() == vertices);
    CHECK(asset.get()->getIndices().data() == indices);
    CHECK(asset.get()->getFaces().data() == faces);
}
