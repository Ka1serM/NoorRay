#include "UI/MaterialXNodes/MaterialXGraphNode.h"

#include "UI/MaterialXNodeCatalog.h"

#include <imgui.h>

#include <vector>

namespace mx = MaterialX;

MaterialXGraphNode::MaterialXGraphNode(mx::NodePtr node)
    : node_(std::move(node))
{
    if (node_)
        setTitle(node_->getName().empty() ? node_->getCategory() : node_->getName());
}

void MaterialXGraphNode::buildPins()
{
    // Declared inputs, not just the authored ones: an input with no pin cannot
    // be connected to, which left most of a surface shader's parameters
    // unreachable in the graph (see exposedInputs).
    for (const mx::InputPtr& input : exposedInputs(node_)) {
        if (!input)
            continue;
        // An input the body draws inline still needs a pin once something is
        // wired into it: the editor syncs connections through the pins, so an
        // input without one would look disconnected and be cleared.
        if (isInlineInput(input->getName()) && !input->getConnectedNode())
            continue;
        addIN<std::string>(input->getName(), {}, ImFlow::ConnectionFilter::None());
    }
    // MaterialX nodes are value-producing elements. A generic output keeps the
    // graph editable even for closure and custom node types whose runtime data
    // type is not known to the UI.
    (void)addOUT<std::string>("out");
}

void MaterialXGraphNode::setMaterialNode(mx::NodePtr node)
{
    node_ = std::move(node);
    if (!node_)
        return;
    setTitle(node_->getName().empty() ? node_->getCategory() : node_->getName());
}

void MaterialXGraphNode::rebuildPins()
{
    std::vector<std::string> inputNames;
    inputNames.reserve(getIns().size());
    for (const std::shared_ptr<ImFlow::Pin>& pin : getIns())
        inputNames.push_back(pin->getName());
    for (const std::string& inputName : inputNames)
        dropIN(inputName.c_str());
    dropOUT("out");
    buildPins();
}

ImFlow::Pin* MaterialXGraphNode::findInputPin(const std::string& inputName)
{
    for (const std::shared_ptr<ImFlow::Pin>& pin : getIns())
        if (pin && pin->getName() == inputName)
            return pin.get();
    return nullptr;
}

void MaterialXGraphNode::draw()
{
    drawBody();
}

void MaterialXGraphNode::drawBody()
{
    if (!node_)
        return;
    ImGui::TextDisabled("%s", node_->getCategory().c_str());
}

bool MaterialXGraphNode::isInlineInput(const std::string&) const
{
    return false;
}

std::string MaterialXGraphNode::inputValue(const std::string& inputName) const
{
    if (!node_)
        return {};
    const mx::InputPtr input = node_->getInput(inputName);
    if (!input || input->getConnectedNode())
        return {};
    return input->getValueString();
}
