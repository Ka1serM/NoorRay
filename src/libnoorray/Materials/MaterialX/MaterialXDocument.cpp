#include "MaterialXDocument.h"

#include <algorithm>
#include <cctype>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>


namespace nr::materialx
{

namespace
{
void addNoorRayExtensions(const MaterialX::DocumentPtr& document)
{
    if (document->getNodeDef("ND_noorray_sellmeier_ior"))
        return;

    const MaterialX::NodeDefPtr definition = document->addNodeDef(
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

MaterialX::NodePtr addDisneyPrincipled(const MaterialX::DocumentPtr& document,
    const std::string& name)
{
    return document->addNode("disney_principled", name, "surfaceshader");
}
} // namespace

MaterialX::DocumentPtr loadStandardLibraries(const std::string& materialXStdlibDir)
{
    MaterialX::DocumentPtr libraries = MaterialX::createDocument();
    const MaterialX::FilePath path(materialXStdlibDir);
    MaterialX::FileSearchPath searchPath;
    searchPath.append(path);
    searchPath.append(path.getParentPath());
    const MaterialX::FilePathVec folders{path.getBaseName()};
    const MaterialX::StringSet loaded = MaterialX::loadLibraries(folders, searchPath, libraries);
    if (loaded.empty())
        throw std::runtime_error(
            "No MaterialX definitions were loaded from " + materialXStdlibDir);
    addNoorRayExtensions(libraries);
    return libraries;
}

MaterialX::DocumentPtr getSharedStandardLibraries()
{
    static const MaterialX::DocumentPtr libraries =
        loadStandardLibraries(NR_MATERIALX_STDLIB_DIR);
    return libraries;
}

MaterialX::DocumentPtr documentFromSvmMaterial(const SvmMaterial& material,
    const AuthoringTexturePathResolver& texturePathResolver)
{
    MaterialX::DocumentPtr document = MaterialX::createDocument();
    const MaterialX::NodePtr principled = addDisneyPrincipled(document,
        "nr_synthetic_disney");
    principled->setInputValue("baseColor",
        MaterialX::Color3(material.albedo.r, material.albedo.g, material.albedo.b));
    principled->setInputValue("metallic", material.metallic);
    principled->setInputValue("specular", material.specular);
    principled->setInputValue("roughness", material.roughness);
    principled->setInputValue("specTrans", material.transmission);
    if (texturePathResolver && material.albedoIndex >= 0) {
        const std::string texturePath = texturePathResolver(material.albedoIndex);
        if (!texturePath.empty()) {
            const MaterialX::NodePtr image = document->addNode(
                "image", "nr_albedo_texture", "color3");
            image->setInputValue("file", texturePath);
            if (const MaterialX::InputPtr file = image->getInput("file"))
                file->setAttribute("colorspace", "srgb_texture");
            principled->setConnectedNode("baseColor", image);
        }
    }
    const MaterialX::NodePtr surfaceMaterial = document->addNode(
        "surfacematerial", "nr_synthetic_material", "material");
    surfaceMaterial->setConnectedNode("surfaceshader", principled);
    document->setDataLibrary(getSharedStandardLibraries());
    return document;
}

MaterialX::DocumentPtr documentFromSvmMaterial(const SvmMaterial& material)
{
    return documentFromSvmMaterial(material, {});
}

MaterialX::DocumentPtr defaultMaterial()
{
    // Use MaterialX's built-in Disney Principled node for the neutral fallback.
    MaterialX::DocumentPtr document = MaterialX::createDocument();
    const MaterialX::NodePtr principled = addDisneyPrincipled(document,
        "nr_default_disney");
    principled->setInputValue("baseColor", MaterialX::Color3(0.8f, 0.8f, 0.8f));
    principled->setInputValue("roughness", 0.5f);
    const MaterialX::NodePtr surfaceMaterial = document->addNode(
        "surfacematerial", "nr_default_material_material", "material");
    surfaceMaterial->setConnectedNode("surfaceshader", principled);
    document->setDataLibrary(getSharedStandardLibraries());
    return document;
}

std::vector<MaterialXImageNode> collectImageNodes(const MaterialX::DocumentPtr& document)
{
    std::vector<MaterialXImageNode> result;
    for (const MaterialX::ElementPtr& elem : document->traverseTree())
    {
        const MaterialX::NodePtr node = elem->asA<MaterialX::Node>();
        if (!node || node->getCategory() != "image")
            continue;
        const MaterialX::InputPtr fileInput = node->getInput("file");
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
