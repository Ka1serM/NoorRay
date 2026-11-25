#include "NoorRay.App/UI/Controllers/NodePanelController.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <string_view>

#include "NoorRay.App/UI/Controllers/NodeSockets.h"
#include "NoorRay.App/UI/Controllers/NodeTreeEditor.h"
#include "NoorRay.App/UI/Controls/FieldCommit.h"
#include "NoorRay.App/UI/Models/AppModel.h"
#include "NoorRay/Materials/NodeCatalog.h"
#include "RmlKit/UI/Models/Model.h"

namespace NoorRay::App {

namespace {

using Materials::Node;
using Materials::NodeCatalog;
using Materials::NodeProperty;
using Materials::NodePropertyKind;
using Materials::NodeTree;

constexpr std::size_t rampStride = 5;
constexpr std::size_t pointStride = 2;
constexpr std::size_t maxRampStops = 32;
constexpr std::size_t maxCurvePoints = 64;
constexpr std::string_view rampElements = "color_ramp.elements";
constexpr std::string_view curvePrefix = "mapping.curve";
constexpr std::string_view imageSetting = "image";

bool isCurve(const std::string_view name) {
    return name.starts_with(curvePrefix) && name != "mapping.curve_count";
}

bool hasImage(const std::string_view type) {
    return type == "ShaderNodeTexImage" || type == "ShaderNodeTexEnvironment";
}

//! A setting row keeps the hidden fields of every editor, and data bindings send change events from them.
bool isSettingEvent(const NodeSettingRow& row, const Rml::Event& event) {
    const Rml::Element* target = event.GetTargetElement();
    if (!target)
        return false;
    if (row.editor == "enum" || row.editor == "image")
        return target->GetTagName() == "select";
    if (target->GetTagName() != "input")
        return false;
    const Rml::String type = target->GetAttribute<Rml::String>("type", "text");
    return row.editor == "toggle" ? type == "checkbox" : type == "text";
}

const NodeProperty* findOwn(const Node& node, const std::string_view name) {
    const auto found = std::ranges::find_if(node.properties, [&](const NodeProperty& property) { return property.name == name; });
    return found == node.properties.end() ? nullptr : &*found;
}

Rml::String gradientOf(const std::span<const float> elements) {
    Rml::String stops;
    for (std::size_t first = 0; first + rampStride <= elements.size(); first += rampStride)
        stops += std::format(", {} {}%", swatchOf({elements[first + 1], elements[first + 2], elements[first + 3], 1.0f}), elements[first] * 100.0f);
    if (elements.size() == rampStride)
        stops += stops;
    return std::format("linear-gradient(to right{})", stops);
}

//! Elements with positions sorted, as Blender keeps the stops of a ramp and the points of a curve.
std::vector<float> sortedBy(const std::vector<float>& values, const std::size_t stride) {
    std::vector<std::size_t> order(values.size() / stride);
    for (std::size_t index = 0; index < order.size(); ++index)
        order[index] = index;
    std::ranges::stable_sort(order, [&](const std::size_t a, const std::size_t b) { return values[a * stride] < values[b * stride]; });
    std::vector<float> sorted;
    sorted.reserve(values.size());
    for (const std::size_t index : order)
        sorted.insert(sorted.end(), values.begin() + static_cast<std::ptrdiff_t>(index * stride),
                      values.begin() + static_cast<std::ptrdiff_t>((index + 1) * stride));
    return sorted;
}

//! A new element in the middle of the widest gap, with the mean of its neighbours.
std::vector<float> withMiddle(std::vector<float> values, const std::size_t stride) {
    const std::size_t count = values.size() / stride;
    if (count < 2) {
        std::vector<float> added(values.begin(), values.end());
        added.insert(added.end(), values.begin(), values.end());
        if (!added.empty())
            added[added.size() - stride] = 1.0f;
        return added;
    }
    std::size_t widest = 0;
    for (std::size_t index = 1; index + 1 < count; ++index)
        if (values[(index + 1) * stride] - values[index * stride] > values[(widest + 1) * stride] - values[widest * stride])
            widest = index;
    std::vector<float> middle(stride);
    for (std::size_t component = 0; component < stride; ++component)
        middle[component] = 0.5f * (values[widest * stride + component] + values[(widest + 1) * stride + component]);
    values.insert(values.begin() + static_cast<std::ptrdiff_t>((widest + 1) * stride), middle.begin(), middle.end());
    return values;
}

}

NodePanelController::NodePanelController(RmlKit::UI::Model& model, AppModel& app, Session::Session& session, NodeTreeEditor& editor)
    : model(model), app(app), session(session), editor(editor) {
    model.bindEvent("nodePanelName", &NodePanelController::onName, this);
    model.bindEvent("nodePanelLabel", &NodePanelController::onLabel, this);
    model.bindEvent("nodePanelValue", &NodePanelController::onValue, this);
    model.bindEvent("nodePanelPick", &NodePanelController::onPick, this);
    model.bindEvent("nodePanelColor", &NodePanelController::onColor, this);
    model.bindEvent("nodePanelSetting", &NodePanelController::onSetting, this);
    model.bindEvent("nodePanelStop", &NodePanelController::onStop, this);
    model.bindEvent("nodePanelStopAdd", &NodePanelController::onStopAdd, this);
    model.bindEvent("nodePanelStopRemove", &NodePanelController::onStopRemove, this);
    model.bindEvent("nodePanelStopPick", &NodePanelController::onStopPick, this);
    model.bindEvent("nodePanelPoint", &NodePanelController::onPoint, this);
    model.bindEvent("nodePanelPointAdd", &NodePanelController::onPointAdd, this);
    model.bindEvent("nodePanelPointRemove", &NodePanelController::onPointRemove, this);
}

const Node* NodePanelController::findNode() const {
    const NodeTree* tree = editor.findShown();
    return tree && node < tree->nodes.size() ? &tree->nodes[node] : nullptr;
}

void NodePanelController::show(const std::uint32_t shown) {
    NodePanelModel& panel = app.nodeGraph.panel;
    if (shown != node) {
        panel.pickerInput = -1;
        panel.pickerSetting = -1;
        panel.pickerStop = -1;
    }
    node = shown;
    panel.settings.clear();
    panel.inputs.clear();
    settingNames.clear();
    const NodeTree* tree = editor.findShown();
    const Node* found = findNode();
    panel.shown = found != nullptr;
    if (!found)
        return;
    const Materials::NodeType* type = NodeCatalog::find(found->type);
    panel.name = found->name;
    panel.label = found->label;
    panel.type = type ? Rml::String(type->name) : Rml::String(found->type);
    showSettings(*found);
    showInputs(*tree, *found);
    if (panel.pickerInput >= static_cast<int>(panel.inputs.size()) || panel.pickerSetting >= static_cast<int>(panel.settings.size()))
        panel.pickerInput = panel.pickerSetting = -1;
}

void NodePanelController::showSettings(const Node& shown) {
    NodePanelModel& panel = app.nodeGraph.panel;
    const Scene::Scene& scene = session.getScene();
    if (hasImage(shown.type)) {
        const NodeProperty* image = findOwn(shown, imageSetting);
        NodeSettingRow row{.label = "Image", .editor = "image", .text = "-1"};
        row.options.push_back({"-1", "None"});
        textures.clear();
        const Core::SlotMap<Scene::Texture>& all = scene.getTextures();
        for (std::uint32_t index = 0; index < all.size(); ++index) {
            const std::string& name = all.getValues()[index].name;
            const std::string file = std::filesystem::path(name).filename().string();
            row.options.push_back({std::to_string(index), file.empty() ? Rml::String(name) : Rml::String(file)});
            textures.push_back(all.keyAt(index));
            if (image && !image->text.empty() && (image->text == name || image->text == file))
                row.text = std::to_string(index);
        }
        panel.settings.push_back(std::move(row));
        settingNames.emplace_back(imageSetting);
    }
    const Materials::NodeType* type = NodeCatalog::find(shown.type);
    if (!type)
        return;
    for (const Materials::NodePropertyType& setting : type->properties) {
        if (setting.value.name == imageSetting || setting.value.name == "mapping.curve_count")
            continue;
        const NodeProperty* own = findOwn(shown, setting.value.name);
        const NodeProperty& value = own ? *own : setting.value;
        NodeSettingRow row{.label = setting.label.empty() ? Rml::String(setting.value.name) : Rml::String(setting.label)};
        switch (value.kind) {
        case NodePropertyKind::Text:
            row.editor = setting.items.empty() ? "text" : "enum";
            row.text = value.text;
            for (const Materials::NodeEnumItem& item : setting.items)
                row.options.push_back({item.identifier, item.name});
            break;
        case NodePropertyKind::Number:
            row.editor = "number";
            row.number = value.number.x;
            break;
        case NodePropertyKind::Integer:
            row.editor = "integer";
            row.number = static_cast<float>(value.integer);
            break;
        case NodePropertyKind::Boolean:
            row.editor = "toggle";
            row.checked = value.integer != 0;
            break;
        case NodePropertyKind::Numbers:
            if (value.name == rampElements) {
                row.editor = "ramp";
                for (std::size_t first = 0; first + rampStride <= value.numbers.size(); first += rampStride)
                    row.stops.push_back({value.numbers[first], value.numbers[first + 4],
                                         swatchOf({value.numbers[first + 1], value.numbers[first + 2], value.numbers[first + 3], 1.0f})});
                row.gradient = gradientOf(value.numbers);
            } else if (isCurve(value.name)) {
                row.editor = "curve";
                for (std::size_t first = 0; first + pointStride <= value.numbers.size(); first += pointStride)
                    row.points.push_back({value.numbers[first], value.numbers[first + 1]});
            }
            break;
        case NodePropertyKind::Texture:
            break;
        }
        if (row.editor.empty())
            continue;
        panel.settings.push_back(std::move(row));
        settingNames.push_back(setting.value.name);
    }
}

void NodePanelController::showInputs(const NodeTree& tree, const Node& shown) {
    std::vector<bool> linked(shown.inputs.size(), false);
    for (const Materials::NodeLink& link : tree.links)
        if (link.toNode == node && link.toSocket < linked.size())
            linked[link.toSocket] = true;
    for (std::uint32_t socket = 0; socket < shown.inputs.size(); ++socket) {
        const Materials::NodeSocket& input = shown.inputs[socket];
        if (!input.enabled || input.hidden || linked[socket] || hidesValue(shown, socket))
            continue;
        NodeGraphSocketRow row = socketRow(input, socket, true);
        if (!row.editor.empty())
            app.nodeGraph.panel.inputs.push_back(std::move(row));
    }
}

NodeProperty NodePanelController::findProperty(const std::size_t setting) const {
    const Node* shown = findNode();
    const std::string& name = settingNames[setting];
    if (const NodeProperty* own = shown ? findOwn(*shown, name) : nullptr)
        return *own;
    const Materials::NodeType* type = shown ? NodeCatalog::find(shown->type) : nullptr;
    if (type)
        for (const Materials::NodePropertyType& entry : type->properties)
            if (entry.value.name == name)
                return entry.value;
    return NodeProperty{.name = name};
}

std::size_t NodePanelController::settingArgument(const Rml::VariantList& arguments, const char* event) const {
    return RmlKit::UI::Model::indexArgument(arguments, event, settingNames.size());
}

std::pair<std::size_t, std::size_t> NodePanelController::pairArguments(const Rml::VariantList& arguments, const char* event, const std::size_t count) const {
    if (arguments.size() != 2)
        RmlKit::UI::fatal(std::format("{} takes two arguments", event));
    const std::size_t first = RmlKit::UI::Model::indexOf(arguments[0], event);
    const std::size_t second = RmlKit::UI::Model::indexOf(arguments[1], event);
    if (first >= count)
        RmlKit::UI::fatal(std::format("{}: the node pane has no row {}", event, first));
    return {first, second};
}

void NodePanelController::report(const std::expected<void, Core::Error>& result) {
    app.nodeGraph.notice = result ? Rml::String() : Rml::String(result.error().message);
    model.dirty("nodeGraph");
}

void NodePanelController::setProperty(NodeProperty property, const std::string& merge) {
    const std::uint32_t edited = node;
    report(editor.edit([&](NodeTree& tree) { return NodeCatalog::setProperty(tree, edited, std::move(property)); }, NodeEdit::Result, merge));
}

void NodePanelController::setNumbers(const std::size_t setting, std::vector<float> numbers, const std::string& merge) {
    NodeProperty property = findProperty(setting);
    property.kind = NodePropertyKind::Numbers;
    property.numbers = std::move(numbers);
    setProperty(std::move(property), merge);
}

void NodePanelController::setInput(const std::uint32_t socket, const Core::float4 value, const std::string& merge) {
    const std::uint32_t edited = node;
    report(editor.edit(
        [&](NodeTree& tree) -> std::expected<void, Core::Error> {
            tree.nodes[edited].inputs[socket].value = value;
            return {};
        },
        NodeEdit::Result, merge));
}

void NodePanelController::pick(const std::uint32_t socket) {
    NodePanelModel& panel = app.nodeGraph.panel;
    const auto found = std::ranges::find_if(panel.inputs, [&](const NodeGraphSocketRow& row) { return row.socket == socket; });
    if (found == panel.inputs.end())
        return;
    const int index = static_cast<int>(found - panel.inputs.begin());
    panel.pickerInput = panel.pickerInput == index ? -1 : index;
    panel.pickerSetting = -1;
    panel.pickerColor = found->swatch;
}

void NodePanelController::onName(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    if (!FieldCommit::isCommit(event))
        return;
    RmlKit::UI::Model::requireNoArguments(arguments, "nodePanelName");
    const NodeTree* tree = editor.findShown();
    const Node* shown = findNode();
    const std::string name = app.nodeGraph.panel.name;
    if (!tree || !shown || name.empty() || name == shown->name)
        return;
    if (std::ranges::any_of(tree->nodes, [&](const Node& other) { return other.name == name; })) {
        app.nodeGraph.notice = std::format("The tree already has a node named {}", name);
        app.nodeGraph.panel.name = shown->name;
        model.dirty("nodeGraph");
        return;
    }
    const std::uint32_t edited = node;
    report(editor.edit([&](NodeTree& changed) { return Materials::NodeCatalog::renameNode(changed, edited, name); }, NodeEdit::Layout));
}

void NodePanelController::onLabel(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    if (!FieldCommit::isCommit(event))
        return;
    RmlKit::UI::Model::requireNoArguments(arguments, "nodePanelLabel");
    const Node* shown = findNode();
    const std::string label = app.nodeGraph.panel.label;
    if (!shown || label == shown->label)
        return;
    const std::uint32_t edited = node;
    report(editor.edit(
        [&](NodeTree& changed) -> std::expected<void, Core::Error> {
            changed.nodes[edited].label = label;
            return {};
        },
        NodeEdit::Layout));
}

void NodePanelController::onValue(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    if (!FieldCommit::isCommit(event))
        return;
    const std::size_t input = RmlKit::UI::Model::indexArgument(arguments, "nodePanelValue", app.nodeGraph.panel.inputs.size());
    const Node* shown = findNode();
    const NodeGraphSocketRow& row = app.nodeGraph.panel.inputs[input];
    if (!shown || !isEditorEvent(row, event))
        return;
    const Core::float4 old = shown->inputs[row.socket].value;
    const Core::float4 value = valueOf(row, old);
    if (value == old)
        return;
    setInput(row.socket, value, std::format("input {} {}", node, row.socket));
}

void NodePanelController::onPick(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    const std::size_t input = RmlKit::UI::Model::indexArgument(arguments, "nodePanelPick", app.nodeGraph.panel.inputs.size());
    pick(app.nodeGraph.panel.inputs[input].socket);
    model.dirty("nodeGraph");
}

void NodePanelController::onStopPick(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    NodePanelModel& panel = app.nodeGraph.panel;
    const auto [setting, stop] = pairArguments(arguments, "nodePanelStopPick", panel.settings.size());
    if (stop >= panel.settings[setting].stops.size())
        return;
    const bool open = panel.pickerSetting == static_cast<int>(setting) && panel.pickerStop == static_cast<int>(stop);
    panel.pickerSetting = open ? -1 : static_cast<int>(setting);
    panel.pickerStop = static_cast<int>(stop);
    panel.pickerInput = -1;
    panel.pickerColor = panel.settings[setting].stops[stop].swatch;
    model.dirty("nodeGraph");
}

void NodePanelController::onColor(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "nodePanelColor");
    const NodePanelModel& panel = app.nodeGraph.panel;
    const std::optional<Core::float3> picked = linearOfSwatch(panel.pickerColor);
    const Node* shown = findNode();
    if (!picked || !shown)
        return;
    if (panel.pickerInput >= 0 && panel.pickerInput < static_cast<int>(panel.inputs.size())) {
        const std::uint32_t socket = panel.inputs[static_cast<std::size_t>(panel.pickerInput)].socket;
        const Core::float4 old = shown->inputs[socket].value;
        setInput(socket, {picked->x, picked->y, picked->z, old.w}, std::format("color {} {}", node, socket));
        return;
    }
    if (panel.pickerSetting < 0 || panel.pickerSetting >= static_cast<int>(settingNames.size()))
        return;
    const auto setting = static_cast<std::size_t>(panel.pickerSetting);
    std::vector<float> elements = findProperty(setting).numbers;
    const std::size_t first = static_cast<std::size_t>(panel.pickerStop) * rampStride;
    if (first + rampStride > elements.size())
        return;
    elements[first + 1] = picked->x;
    elements[first + 2] = picked->y;
    elements[first + 3] = picked->z;
    setNumbers(setting, std::move(elements), std::format("stop {} {} {}", node, setting, panel.pickerStop));
}

void NodePanelController::onSetting(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    if (!FieldCommit::isCommit(event))
        return;
    const std::size_t setting = settingArgument(arguments, "nodePanelSetting");
    const NodeSettingRow& row = app.nodeGraph.panel.settings[setting];
    if (!isSettingEvent(row, event))
        return;
    if (row.editor == "image") {
        const int index = std::atoi(row.text.c_str());
        const std::uint32_t edited = node;
        if (index < 0 || static_cast<std::size_t>(index) >= textures.size()) {
            setProperty(NodeProperty{.name = std::string(imageSetting), .kind = NodePropertyKind::Text}, {});
            return;
        }
        const Scene::TextureHandle texture = textures[static_cast<std::size_t>(index)];
        report(editor.edit([&](NodeTree& tree) { return NodeCatalog::setTexture(tree, edited, session.getScene(), texture); }, NodeEdit::Result));
        return;
    }
    NodeProperty property = findProperty(setting);
    const NodeProperty old = property;
    if (row.editor == "enum" || row.editor == "text")
        property.text = row.text;
    else if (row.editor == "number")
        property.number.x = row.number;
    else if (row.editor == "integer")
        property.integer = std::lround(float(row.number));
    else if (row.editor == "toggle")
        property.integer = row.checked ? 1 : 0;
    if (property == old)
        return;
    setProperty(std::move(property), std::format("setting {} {}", node, setting));
}

void NodePanelController::onStop(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    if (!FieldCommit::isCommit(event))
        return;
    const auto [setting, stop] = pairArguments(arguments, "nodePanelStop", settingNames.size());
    const NodeSettingRow& row = app.nodeGraph.panel.settings[setting];
    std::vector<float> elements = findProperty(setting).numbers;
    const std::size_t first = stop * rampStride;
    if (stop >= row.stops.size() || first + rampStride > elements.size())
        return;
    elements[first] = std::clamp<float>(row.stops[stop].position, 0.0f, 1.0f);
    elements[first + 4] = std::clamp<float>(row.stops[stop].alpha, 0.0f, 1.0f);
    setNumbers(setting, sortedBy(elements, rampStride), std::format("stop {} {} {}", node, setting, stop));
}

void NodePanelController::onStopAdd(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    const std::size_t setting = settingArgument(arguments, "nodePanelStopAdd");
    const std::vector<float> elements = findProperty(setting).numbers;
    if (elements.size() / rampStride >= maxRampStops)
        return;
    setNumbers(setting, withMiddle(elements, rampStride), {});
}

void NodePanelController::onStopRemove(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    const auto [setting, stop] = pairArguments(arguments, "nodePanelStopRemove", settingNames.size());
    std::vector<float> elements = findProperty(setting).numbers;
    if (elements.size() <= rampStride || (stop + 1) * rampStride > elements.size())
        return;
    elements.erase(elements.begin() + static_cast<std::ptrdiff_t>(stop * rampStride), elements.begin() + static_cast<std::ptrdiff_t>((stop + 1) * rampStride));
    app.nodeGraph.panel.pickerSetting = -1;
    setNumbers(setting, std::move(elements), {});
}

void NodePanelController::onPoint(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    if (!FieldCommit::isCommit(event))
        return;
    const auto [setting, point] = pairArguments(arguments, "nodePanelPoint", settingNames.size());
    const NodeSettingRow& row = app.nodeGraph.panel.settings[setting];
    std::vector<float> points = findProperty(setting).numbers;
    const std::size_t first = point * pointStride;
    if (point >= row.points.size() || first + pointStride > points.size())
        return;
    points[first] = row.points[point].x;
    points[first + 1] = row.points[point].y;
    setNumbers(setting, sortedBy(points, pointStride), std::format("point {} {} {}", node, setting, point));
}

void NodePanelController::onPointAdd(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    const std::size_t setting = settingArgument(arguments, "nodePanelPointAdd");
    const std::vector<float> points = findProperty(setting).numbers;
    if (points.size() / pointStride >= maxCurvePoints)
        return;
    setNumbers(setting, withMiddle(points, pointStride), {});
}

void NodePanelController::onPointRemove(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    const auto [setting, point] = pairArguments(arguments, "nodePanelPointRemove", settingNames.size());
    std::vector<float> points = findProperty(setting).numbers;
    if (points.size() <= 2 * pointStride || (point + 1) * pointStride > points.size())
        return;
    points.erase(points.begin() + static_cast<std::ptrdiff_t>(point * pointStride), points.begin() + static_cast<std::ptrdiff_t>((point + 1) * pointStride));
    setNumbers(setting, std::move(points), {});
}

}
