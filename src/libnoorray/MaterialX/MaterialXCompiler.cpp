#include "MaterialXCompiler.h"

#include <algorithm>
#include <cctype>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>

namespace mx = MaterialX;

namespace nr::materialx
{

mx::DocumentPtr loadStandardLibraries(const std::string& materialXStdlibDir)
{
    mx::DocumentPtr libraries = mx::createDocument();
    const mx::FilePath path(materialXStdlibDir);
    mx::FileSearchPath searchPath;
    searchPath.append(path);
    searchPath.append(path.getParentPath());
    const mx::FilePathVec folders{path.getBaseName()};
    const mx::StringSet loaded = mx::loadLibraries(folders, searchPath, libraries);
    if (loaded.empty())
        throw std::runtime_error(
            "No MaterialX definitions were loaded from " + materialXStdlibDir);
    return libraries;
}

mx::DocumentPtr getSharedStandardLibraries()
{
    static const mx::DocumentPtr libraries =
        loadStandardLibraries(NR_MATERIALX_STDLIB_DIR);
    return libraries;
}

mx::DocumentPtr documentFromAuthoring(const MaterialAuthoring& material)
{
    mx::DocumentPtr document = mx::createDocument();
    const mx::NodePtr shader = document->addNode(
        "open_pbr_surface", "nr_synthetic_open_pbr", "surfaceshader");
    shader->setInputValue("base_weight",
        material.albedo != glm::vec3(0.0f) ? 1.0f : 0.0f);
    shader->setInputValue("base_color",
        mx::Color3(material.albedo.r, material.albedo.g, material.albedo.b));
    shader->setInputValue("base_metalness", material.metallic);
    shader->setInputValue("specular_weight", material.specular);
    shader->setInputValue("specular_roughness", material.roughness);
    shader->setInputValue("transmission_weight", material.transmission);
    shader->setInputValue("transmission_color", mx::Color3(
        material.transmissionColor.r, material.transmissionColor.g,
        material.transmissionColor.b));
    shader->setInputValue("geometry_opacity", material.opacity);
    shader->setInputValue("emission_color", mx::Color3(
        material.emission.r, material.emission.g, material.emission.b));
    shader->setInputValue("emission_luminance", material.emissionStrength);
    const mx::NodePtr surfaceMaterial = document->addNode(
        "surfacematerial", "nr_synthetic_material", "material");
    surfaceMaterial->setConnectedNode("surfaceshader", shader);
    return document;
}

mx::DocumentPtr defaultMaterial()
{
    // The default "no material authored" document. A flat medium grey
    // (albedo 0.8, roughness 0.5) matches the native grey fallback shared by
    // the importers and the Hydra plugin, so an unassigned material always
    // reads as grey rather than black or a white mirror.
    mx::DocumentPtr document = mx::createDocument();
    const mx::NodePtr shader = document->addNode(
        "open_pbr_surface", "nr_default_material", "surfaceshader");
    shader->setInputValue("base_color", mx::Color3(0.8f, 0.8f, 0.8f));
    shader->setInputValue("specular_roughness", 0.5f);
    const mx::NodePtr surfaceMaterial = document->addNode(
        "surfacematerial", "nr_default_material_material", "material");
    surfaceMaterial->setConnectedNode("surfaceshader", shader);
    return document;
}

std::vector<MaterialXImageNode> collectImageNodes(const mx::DocumentPtr& document)
{
    std::vector<MaterialXImageNode> result;
    for (const mx::ElementPtr& elem : document->traverseTree())
    {
        const mx::NodePtr node = elem->asA<mx::Node>();
        if (!node || node->getCategory() != "image")
            continue;
        const mx::InputPtr fileInput = node->getInput("file");
        if (!fileInput)
            continue;
        const std::string raw = fileInput->getValueString();
        if (raw.empty())
            continue;
        const std::string& type = node->getType();
        std::string colorSpace = node->getActiveColorSpace();
        std::ranges::transform(colorSpace, colorSpace.begin(),
            [](const unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
        const bool srgb = !colorSpace.empty()
            ? colorSpace.find("srgb") != std::string::npos
            : type == "color3" || type == "color4";
        result.push_back({raw, srgb
                ? MaterialXImageColorSpace::Srgb
                : MaterialXImageColorSpace::Linear});
    }
    return result;
}

} // namespace nr::materialx