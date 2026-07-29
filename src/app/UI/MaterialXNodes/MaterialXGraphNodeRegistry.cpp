#include "UI/MaterialXNodes/MaterialXGraphNodeRegistry.h"

namespace mx = MaterialX;

MaterialXGraphNodeRegistry& MaterialXGraphNodeRegistry::instance()
{
    // Function-local so registrars running during static initialisation in
    // other translation units always find a constructed registry.
    static MaterialXGraphNodeRegistry registry;
    return registry;
}

void MaterialXGraphNodeRegistry::registerCategory(
    const std::string& category, MaterialXGraphNodeCreator creator)
{
    byCategory_[category] = std::move(creator);
}

void MaterialXGraphNodeRegistry::registerOutputType(
    const std::string& outputType, MaterialXGraphNodeCreator creator)
{
    byOutputType_[outputType] = std::move(creator);
}

const MaterialXGraphNodeCreator* MaterialXGraphNodeRegistry::find(
    const std::string& category, const std::string& outputType) const
{
    if (const auto found = byCategory_.find(category); found != byCategory_.end())
        return &found->second;
    if (const auto found = byOutputType_.find(outputType); found != byOutputType_.end())
        return &found->second;
    return nullptr;
}

std::shared_ptr<MaterialXGraphNode> addMaterialXGraphNode(
    ImFlow::ImNodeFlow& flow, const ImVec2& position, mx::NodePtr node)
{
    const MaterialXGraphNodeCreator* creator =
        MaterialXGraphNodeRegistry::instance().find(node->getCategory(), node->getType());
    std::shared_ptr<MaterialXGraphNode> created = creator
        ? (*creator)(flow, position, std::move(node))
        : flow.addNode<MaterialXGraphNode>(position, std::move(node));
    created->buildPins();
    return created;
}
