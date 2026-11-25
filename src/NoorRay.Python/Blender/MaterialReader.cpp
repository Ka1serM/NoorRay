#include "NoorRay.Python/Blender/MaterialReader.h"

#include <utility>

#include "NoorRay/Materials/Nodes/NodeAccess.h"

namespace NoorRay::Python {

MaterialReader::Result MaterialReader::read(const Object& material) {
    Result result;
    const std::string name = reader.text(material, "name");
    const Object source = nodeTree(reader, material);
    std::shared_ptr<Materials::NodeTree> tree = trees.read(source);
    std::vector<Materials::NodeProperty>& settings = tree->properties;
    settings.push_back(booleanProperty("use_nodes", static_cast<bool>(source)));
    settings.push_back(integerProperty("pass_index", reader.integer(material, "pass_index")));
    const Object cycles = reader.find(material, "cycles");
    const Object method = reader.find(material, "displacement_method");
    settings.push_back(textProperty("displacement_method", method ? reader.text(method) : cycles ? reader.text(cycles, "displacement_method") : std::string()));
    settings.push_back(textProperty("emission_sampling", cycles ? reader.text(cycles, "emission_sampling") : std::string()));
    const Values color = reader.values(material, "diffuse_color");
    settings.push_back(numbersProperty("diffuse_color", std::span(color.values).first(color.count)));
    settings.push_back(numberProperty("metallic", reader.number(material, "metallic")));
    settings.push_back(numberProperty("roughness", reader.number(material, "roughness")));
    BlenderTextures loader(reader, diagnostics, textures, scene, trees, images);
    if (const Materials::NodeRef output = source ? Materials::findOutput(*tree, "ShaderNodeOutputMaterial") : Materials::NodeRef()) {
        Materials::NodeLinks links;
        if (const Materials::NodeLinkEnd volume = links.linkedNode(Materials::inputNamed(output, "Volume")); volume.node)
            result.volume = volumes.read(volume.node, name, nullptr, loader);
    }
    Materials::MaterialCompiler compiler(diagnostics.forNodes(), loader, &attributeTables, folder);
    Materials::CompiledMaterial compiled = compiler.compile(*tree, name);
    diagnostics.settle();
    loader.storeRecords();
    for (Materials::NodeProperty& record : Materials::attributeRecords(compiled.needs.streams, attributeTables.instances))
        settings.push_back(std::move(record));
    result.material = std::move(compiled.material);
    result.material.nodeTree = std::move(tree);
    result.needs = std::move(compiled.needs);
    return result;
}

std::optional<Scene::Volume> MaterialReader::readVolume(const Materials::NodeTree& tree, const std::string& name, const char* outputType, GridSet* grids) {
    const Materials::NodeRef output = Materials::findOutput(tree, outputType);
    Materials::NodeLinks links;
    const Materials::NodeLinkEnd link = output ? links.linkedNode(Materials::inputNamed(output, "Volume")) : Materials::NodeLinkEnd{};
    if (!link.node)
        return std::nullopt;
    BlenderTextures loader(reader, diagnostics, textures, scene, trees, images);
    return volumes.read(link.node, name, grids, loader);
}

}
