#include "MaterialXDocument.h"

#include <algorithm>
#include <cctype>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>

namespace mx = MaterialX;

namespace nr::materialx
{

namespace
{
void addNoorRayExtensions(const mx::DocumentPtr& document)
{
    if (document->getNodeDef("ND_noorray_sellmeier_ior"))
        return;

    const mx::NodeDefPtr definition = document->addNodeDef(
        "ND_noorray_sellmeier_ior", "float", "noorray_sellmeier_ior");
    definition->setNodeGroup("spectral");
    definition->setAttribute("doc",
        "Sellmeier dispersion model; scalar consumers use 546.074 nm while NoorRay evaluates sampled wavelengths");
    const auto addInput = [&](const char* name, const char* value) {
        definition->addInput(name, "float")->setValueString(value);
    };
    addInput("b1", "1.03961212");
    addInput("b2", "0.231792344");
    addInput("b3", "1.01046945");
    addInput("c1", "0.00600069867");
    addInput("c2", "0.0200179144");
    addInput("c3", "103.560653");
}

mx::NodePtr addDisneyPrincipled(const mx::DocumentPtr& document,
    const std::string& name)
{
    return document->addNode("disney_principled", name, "surfaceshader");
}
} // namespace

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
    addNoorRayExtensions(libraries);
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
    const mx::NodePtr principled = addDisneyPrincipled(document,
        "nr_synthetic_disney");
    principled->setInputValue("baseColor",
        mx::Color3(material.albedo.r, material.albedo.g, material.albedo.b));
    principled->setInputValue("metallic", material.metallic);
    principled->setInputValue("specular", material.specular);
    principled->setInputValue("roughness", material.roughness);
    principled->setInputValue("specTrans", material.transmission);
    const mx::NodePtr surfaceMaterial = document->addNode(
        "surfacematerial", "nr_synthetic_material", "material");
    surfaceMaterial->setConnectedNode("surfaceshader", principled);
    document->setDataLibrary(getSharedStandardLibraries());
    return document;
}

mx::DocumentPtr defaultMaterial()
{
    // Use MaterialX's built-in Disney Principled node for the neutral fallback.
    mx::DocumentPtr document = mx::createDocument();
    const mx::NodePtr principled = addDisneyPrincipled(document,
        "nr_default_disney");
    principled->setInputValue("baseColor", mx::Color3(0.8f, 0.8f, 0.8f));
    principled->setInputValue("roughness", 0.5f);
    const mx::NodePtr surfaceMaterial = document->addNode(
        "surfacematerial", "nr_default_material_material", "material");
    surfaceMaterial->setConnectedNode("surfaceshader", principled);
    document->setDataLibrary(getSharedStandardLibraries());
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
        // MaterialX image color space is authored on the filename input. A
        // document-level color space is only the fallback when that input has
        // no explicit colorspace override.
        std::string colorSpace = fileInput->getAttribute("colorspace");
        if (colorSpace.empty())
            colorSpace = node->getActiveColorSpace();
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
