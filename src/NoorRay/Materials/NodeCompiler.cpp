#include "NoorRay/Materials/NodeCompiler.h"

#include <format>
#include <memory>

#include "NoorRay/Materials/NodeTree.h"
#include "NoorRay/Materials/Nodes/Folder.h"
#include "NoorRay/Materials/Nodes/LightCompiler.h"
#include "NoorRay/Materials/Nodes/MaterialCompiler.h"
#include "NoorRay/Materials/Nodes/NodeAttributes.h"
#include "NoorRay/Materials/Nodes/NodeDiagnostics.h"
#include "NoorRay/Materials/Nodes/NodeTextures.h"

namespace NoorRay::Materials {

namespace {

std::expected<std::vector<std::string>, Core::Error> finish(NodeDiagnostics& diagnostics) {
    if (diagnostics.failed())
        return Core::fail(Core::ErrorCode::InvalidMaterial, *diagnostics.getError());
    return diagnostics.takeWarnings();
}

}

std::expected<std::vector<std::string>, Core::Error> NodeCompiler::compileMaterial(Scene::Scene& scene, const Scene::MaterialHandle handle,
                                                                                   const bool approximate) {
    const Scene::Material* found = scene.find(handle);
    if (!found)
        return Core::fail(Core::ErrorCode::InvalidArgument, "the material does not exist");
    if (!found->nodeTree)
        return Core::fail(Core::ErrorCode::InvalidMaterial, std::format("material '{}' has no node tree", found->name));
    const std::shared_ptr<const NodeTree> tree = found->nodeTree;
    NodeDiagnostics diagnostics(approximate);
    RecordedTextures textures;
    AttributeTables tables;
    seedAttributes(tree->properties, tables);
    //! The folder starts a new random sequence, so a chain of procedural textures can average to a value a little apart from the one of a sync.
    Folder folder;
    MaterialCompiler compiler(diagnostics, textures, &tables, folder);
    CompiledMaterial compiled = compiler.compile(*tree, found->name);
    if (diagnostics.failed())
        return finish(diagnostics);
    compiled.material.nodeTree = tree;
    scene.setMaterial(handle, compiled.material);
    return finish(diagnostics);
}

std::expected<std::vector<std::string>, Core::Error> NodeCompiler::compileLight(Scene::Scene& scene, const Scene::LightHandle handle, const bool approximate) {
    const Scene::Light* found = scene.find(handle);
    if (!found)
        return Core::fail(Core::ErrorCode::InvalidArgument, "the light does not exist");
    if (!found->nodeTree)
        return Core::fail(Core::ErrorCode::InvalidMaterial, std::format("light '{}' has no node tree", found->name));
    Scene::Light light = *found;
    NodeDiagnostics diagnostics(approximate);
    RecordedTextures textures;
    Materials::compileLight(*light.nodeTree, light, diagnostics, textures);
    if (diagnostics.failed())
        return finish(diagnostics);
    scene.setLight(handle, light);
    return finish(diagnostics);
}

std::expected<std::vector<std::string>, Core::Error> NodeCompiler::compileEnvironment(Scene::Scene& scene, const bool approximate) {
    const Scene::Environment& last = scene.getEnvironment();
    if (!last.nodeTree)
        return Core::fail(Core::ErrorCode::InvalidMaterial, "the environment has no node tree");
    const std::shared_ptr<const NodeTree> tree = last.nodeTree;
    NodeDiagnostics diagnostics(approximate);
    RecordedTextures textures;
    Scene::Environment environment = Materials::compileEnvironment(*tree, last, diagnostics, textures);
    if (diagnostics.failed())
        return finish(diagnostics);
    environment.nodeTree = tree;
    scene.setEnvironment(environment);
    return finish(diagnostics);
}

}
