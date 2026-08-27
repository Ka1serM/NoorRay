#include "NoorRaySession.h"
#include "Materials/MaterialX/MaterialXDocument.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Scene/Objects/MeshInstance.h"
#include "Scene/Scene.h"
#include "Scene/Import/SceneReader.h"
#include "Scene/Import/SceneWriter.h"
#include "Materials/SVM/SvmCompiler.h"

#include <MaterialXCore/Document.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

TEST_CASE("default MaterialX material compiles with its own libraries",
    "[materialx][default]")
{
    const auto document = nr::materialx::defaultMaterial();
    nr::svm::SvmCompiler compiler;
    const auto program = compiler.compile(document);
    REQUIRE_FALSE(program.bytecode.empty());
}

TEST_CASE("USD scene round-trip preserves geometry and embedded MaterialX", "[scene][usd][materialx]")
{
    noorray::NoorRaySession session;
    Scene& scene = session.scene;
    const auto material = nr::materialx::defaultMaterial();
    const auto asset = scene.add(MeshAsset::CreateSphere(scene, "RoundTripSphere", material, 8, 16));
    std::vector<Vertex> authoredVertices(
        asset.get()->getVertices().begin(), asset.get()->getVertices().end());
    authoredVertices[1].color = nr::vertex_color::packLinear(
        glm::vec4(0.2f, 0.4f, 0.8f, 0.6f));
    authoredVertices[1].tangent = glm::vec3(0.0f, 0.0f, 1.0f);
    authoredVertices[1].tangentSign = -1.0f;
    asset.get()->updateVertexData(authoredVertices);
    scene.add(std::make_unique<MeshInstance>(scene, "RoundTripSphere", asset, Transform{}));

    const std::filesystem::path path = std::filesystem::temp_directory_path() / "noorray-usd-roundtrip.usda";
    SceneWriter::Write(scene, path.string());
    REQUIRE(std::filesystem::is_regular_file(path));

    Scene reloaded;
    SceneReader::Read(reloaded, path.string());
    REQUIRE(reloaded.getMeshInstances().size() == 1);
    const MeshAsset& reloadedAsset = reloaded.getMeshInstances().front()->getMeshAsset();
    REQUIRE(reloadedAsset.getVertices().size() == asset.get()->getVertices().size());
    REQUIRE(reloadedAsset.getIndices().size() == asset.get()->getIndices().size());
    REQUIRE(reloadedAsset.getFaces().size() == asset.get()->getFaces().size());
    for (size_t i = 0; i < asset.get()->getVertices().size(); ++i) {
        const Vertex& expected = asset.get()->getVertices()[i];
        const Vertex& actual = reloadedAsset.getVertices()[i];
        for (int component = 0; component < 3; ++component) {
            REQUIRE(actual.position[component] == expected.position[component]);
            REQUIRE(actual.normal[component] == expected.normal[component]);
            REQUIRE(actual.tangent[component] == expected.tangent[component]);
        }
        REQUIRE(actual.tangentSign == expected.tangentSign);
        REQUIRE(actual.uv.x == expected.uv.x);
        REQUIRE(actual.uv.y == expected.uv.y);
        REQUIRE(actual.color == expected.color);
    }
    for (size_t i = 0; i < asset.get()->getIndices().size(); ++i)
        REQUIRE(reloadedAsset.getIndices()[i] == asset.get()->getIndices()[i]);
    for (size_t i = 0; i < asset.get()->getFaces().size(); ++i)
        REQUIRE(reloadedAsset.getFaces()[i].materialIndex
            == asset.get()->getFaces()[i].materialIndex);
    REQUIRE(reloadedAsset.getMaterialCount() == 1);
    REQUIRE(reloaded.getMaterialXDocuments().size() > reloadedAsset.getMaterialIds()[0]);
    REQUIRE(reloaded.getMaterialXDocuments()[reloadedAsset.getMaterialIds()[0]]);

    // USD stores the MaterialX XML, not the document's in-memory data-library
    // pointer. A reopened document must resolve its standard nodedefs again
    // before the SVM compiler sees it.
    const auto reloadedMaterial =
        reloaded.getMaterialXDocuments()[reloadedAsset.getMaterialIds()[0]];
    reloadedMaterial->setDataLibrary(nr::materialx::getSharedStandardLibraries());
    nr::svm::SvmCompiler compiler;
    REQUIRE_FALSE(compiler.compile(reloadedMaterial).bytecode.empty());

    std::filesystem::remove(path);
}
