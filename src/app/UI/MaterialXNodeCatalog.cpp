#include "MaterialXNodeCatalog.h"

#include "Materials/MaterialX/MaterialXDocument.h"

#include <MaterialXCore/Definition.h>

#include <algorithm>
#include <cctype>
#include <sstream>

namespace mx = MaterialX;

namespace
{
std::string toLower(std::string text)
{
    std::ranges::transform(text, text.begin(),
        [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

// Definitions without a nodegroup still need a heading. Bucket them by what
// they produce, which is how the MaterialX documentation groups them too.
std::string groupForOutputType(const std::string& outputType)
{
    if (outputType == "surfaceshader" || outputType == "displacementshader"
        || outputType == "volumeshader" || outputType == "lightshader")
        return "shader";
    if (outputType == "BSDF" || outputType == "EDF" || outputType == "VDF")
        return "pbr";
    if (outputType == "material")
        return "material";
    return "other";
}

std::vector<std::string> splitTerms(const std::string& query)
{
    std::vector<std::string> terms;
    std::istringstream stream(toLower(query));
    for (std::string term; stream >> term;)
        terms.push_back(term);
    return terms;
}
}

MaterialXNodeCatalog::MaterialXNodeCatalog()
{
    try {
        libraries_ = nr::materialx::loadStandardLibraries(NR_MATERIALX_STDLIB_DIR);
    }
    catch (const std::exception& error) {
        loadError_ = error.what();
        return;
    }

    for (const mx::NodeDefPtr& definition : libraries_->getNodeDefs()) {
        if (!definition || definition->getNodeString().empty())
            continue;
        MaterialXNodeType type;
        type.category = definition->getNodeString();
        type.nodeDefName = definition->getName();
        type.outputType = definition->getType();
        type.group = definition->getNodeGroup();
        if (type.group.empty())
            type.group = groupForOutputType(type.outputType);
        // Keep the original MaterialX category visible in the menu. Search
        // aliases below make constants discoverable as values/colors/vectors
        // without renaming the actual node users author in MaterialX.
        type.label = type.category + " (" + type.outputType + ")";
        const std::string canonicalName = "ND_" + type.category + "_" + type.outputType;
        if (type.nodeDefName != canonicalName)
            type.label += " [" + type.nodeDefName + "]";
        type.documentation = definition->getAttribute("doc");
        type.searchKey = toLower(type.category + " " + type.group + " "
            + type.outputType + " " + type.nodeDefName);
        if (type.category == "constant") {
            type.searchKey += " value";
            if (type.outputType == "float")
                type.searchKey += " scalar";
            if (type.outputType.starts_with("color"))
                type.searchKey += " color";
            if (type.outputType.starts_with("vector"))
                type.searchKey += " vector";
        }
        types_.push_back(std::move(type));
    }

    std::ranges::sort(types_, [](const MaterialXNodeType& a, const MaterialXNodeType& b) {
        if (a.group != b.group)
            return a.group < b.group;
        if (a.category != b.category)
            return a.category < b.category;
        return a.outputType < b.outputType;
    });

    for (const MaterialXNodeType& type : types_)
        if (groups_.empty() || groups_.back() != type.group)
            groups_.push_back(type.group);
}

const MaterialXNodeCatalog& MaterialXNodeCatalog::instance()
{
    static const MaterialXNodeCatalog catalog;
    return catalog;
}

std::vector<const MaterialXNodeType*> MaterialXNodeCatalog::search(
    const std::string& query) const
{
    const std::vector<std::string> terms = splitTerms(query);
    std::vector<const MaterialXNodeType*> results;
    for (const MaterialXNodeType& type : types_) {
        const bool matches = std::ranges::all_of(terms,
            [&type](const std::string& term) {
                return type.searchKey.find(term) != std::string::npos;
            });
        if (matches)
            results.push_back(&type);
    }
    return results;
}

std::vector<const MaterialXNodeType*> MaterialXNodeCatalog::inGroup(
    const std::string& group) const
{
    std::vector<const MaterialXNodeType*> results;
    for (const MaterialXNodeType& type : types_)
        if (type.group == group)
            results.push_back(&type);
    return results;
}

mx::ConstNodeDefPtr MaterialXNodeCatalog::findNodeDef(const MaterialXNodeType& type) const
{
    return libraries_ ? libraries_->getNodeDef(type.nodeDefName) : nullptr;
}

std::vector<mx::InputPtr> exposedInputs(const mx::NodePtr& node)
{
    std::vector<mx::InputPtr> result;
    if (!node)
        return result;

    if (const mx::ConstNodeDefPtr declaration = MaterialXNodeCatalog::instance()
            .findNodeDef(node->getCategory(), node->getType()))
        result = declaration->getActiveInputs();

    // A hand-authored document, or a definition the catalog failed to load,
    // can carry inputs the declaration has never heard of. Dropping those
    // would hide parts of a document the editor is about to write back.
    for (const mx::InputPtr& authored : node->getInputs()) {
        if (!authored)
            continue;
        const bool declared = std::ranges::any_of(result,
            [&authored](const mx::InputPtr& candidate) {
                return candidate && candidate->getName() == authored->getName();
            });
        if (!declared)
            result.push_back(authored);
    }
    return result;
}

mx::ConstNodeDefPtr MaterialXNodeCatalog::findNodeDef(
    const std::string& category, const std::string& outputType) const
{
    if (!libraries_)
        return nullptr;
    const MaterialXNodeType* fallback = nullptr;
    for (const MaterialXNodeType& type : types_) {
        if (type.category != category)
            continue;
        if (type.outputType == outputType)
            return libraries_->getNodeDef(type.nodeDefName);
        // Surface shaders are declared as "surfaceshader" while the node in
        // the document may carry a more specific type string; keep the first
        // same-category declaration so those still resolve.
        if (!fallback)
            fallback = &type;
    }
    return fallback ? libraries_->getNodeDef(fallback->nodeDefName) : nullptr;
}
