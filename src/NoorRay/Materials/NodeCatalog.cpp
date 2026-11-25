#include "NoorRay/Materials/NodeCatalog.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <format>
#include <numeric>
#include <optional>
#include <ranges>
#include <unordered_set>
#include <utility>

#include "NoorRay/Materials/Nodes/CurveSampler.h"
#include "NoorRay/Materials/Nodes/NodeAccess.h"
#include "NoorRay/Materials/Nodes/NodeCatalogData.h"
#include "NoorRay/Materials/Nodes/NodeImage.h"
#include "NoorRay/Materials/Nodes/NodeTextures.h"
#include "NoorRay/Materials/Nodes/RampSampler.h"

namespace NoorRay::Materials {

namespace {

constexpr std::string_view rerouteType = "NodeReroute";
constexpr std::string_view activeOutput = "is_active_output";
constexpr std::string_view hideValue = "hide_value";
constexpr std::string_view internalLinks = "internal_links";
constexpr std::string_view rampElements = "color_ramp.elements";
constexpr std::string_view rampSamples = "color_ramp.samples";

NodeSocket toSocket(const NodeSocketData& data) {
    return {std::string(data.identifier), std::string(data.name), data.type, data.components, data.enabled, data.hidden, data.value, std::string(data.text)};
}

NodeProperty toProperty(const NodePropertyData& data) {
    NodeProperty property;
    property.name = data.name;
    property.kind = data.kind;
    property.number.x = data.number;
    property.integer = data.integer;
    property.text = data.text;
    property.numbers.assign(data.numbers.begin(), data.numbers.end());
    return property;
}

std::vector<NodeSocketTemplate> toTemplates(const std::span<const NodeSocketData> sockets) {
    std::vector<NodeSocketTemplate> templates;
    templates.reserve(sockets.size());
    for (const NodeSocketData& data : sockets)
        templates.push_back({toSocket(data), data.minimum, data.maximum, std::string(data.subtype), data.hideValue});
    return templates;
}

NodeType toType(const NodeTypeData& data) {
    NodeType type;
    type.type = data.type;
    type.name = data.name;
    type.category = data.category;
    type.width = data.width;
    type.inputs = toTemplates(data.inputs);
    type.outputs = toTemplates(data.outputs);
    type.trees.assign(data.trees.begin(), data.trees.end());
    for (const NodePropertyData& property : data.properties) {
        if (!property.shown)
            continue;
        NodePropertyType& added = type.properties.emplace_back();
        added.value = toProperty(property);
        added.label = property.label;
        added.minimum = property.minimum;
        added.maximum = property.maximum;
        for (const NodeEnumData& item : property.items)
            added.items.push_back({std::string(item.identifier), std::string(item.name)});
    }
    return type;
}

NodeProperty* findMutable(std::vector<NodeProperty>& properties, const std::string_view name) {
    const auto found = std::ranges::find(properties, name, &NodeProperty::name);
    return found == properties.end() ? nullptr : &*found;
}

void eraseProperty(std::vector<NodeProperty>& properties, const std::string_view name) {
    std::erase_if(properties, [&](const NodeProperty& property) { return property.name == name; });
}

NodeProperty numbersProperty(const std::string_view name, std::vector<float> numbers) {
    NodeProperty property;
    property.name = name;
    property.kind = NodePropertyKind::Numbers;
    property.numbers = std::move(numbers);
    return property;
}

void storeProperty(std::vector<NodeProperty>& properties, NodeProperty property) {
    if (NodeProperty* existing = findMutable(properties, property.name))
        *existing = std::move(property);
    else
        properties.push_back(std::move(property));
}

const NodePropertyData* findData(const NodeTypeData& type, const std::string_view name) {
    const auto found = std::ranges::find(type.properties, name, &NodePropertyData::name);
    return found == type.properties.end() ? nullptr : &*found;
}

//! The index of the value of a rule setting among its choices.
std::size_t choiceOf(const Node& node, const NodePropertyData& setting) {
    const NodeProperty* value = findProperty(node.properties, setting.name);
    if (setting.kind == NodePropertyKind::Boolean)
        return (value ? value->integer : setting.integer) != 0 ? 1 : 0;
    const std::string_view identifier = value ? std::string_view(value->text) : setting.text;
    const auto found = std::ranges::find(setting.items, identifier, &NodeEnumData::identifier);
    return found == setting.items.end() ? 0 : static_cast<std::size_t>(found - setting.items.begin());
}

void setEnabled(std::vector<NodeSocket>& sockets, const std::span<const NodeSocketData> data, const std::uint64_t enabled) {
    for (std::size_t index = 0; index < data.size(); ++index) {
        const auto socket = std::ranges::find(sockets, data[index].identifier, &NodeSocket::identifier);
        if (socket == sockets.end())
            continue;
        socket->enabled = (enabled >> index & 1) != 0;
        socket->name = data[index].name;
    }
}

//! Gives the sockets the availability and names that the settings of the node select.
void applyRules(Node& node, const NodeTypeData& type) {
    if (type.ruleSettings.empty())
        return;
    std::size_t combination = 0;
    for (const std::string_view name : type.ruleSettings) {
        const NodePropertyData& setting = *findData(type, name);
        const std::size_t choices = setting.kind == NodePropertyKind::Boolean ? 2 : setting.items.size();
        combination = combination * choices + choiceOf(node, setting);
    }
    const NodeSocketState& state = type.states[type.ruleStates[combination]];
    setEnabled(node.inputs, type.inputs, state.enabledInputs);
    setEnabled(node.outputs, type.outputs, state.enabledOutputs);
    for (const NodeSocketName& named : state.names) {
        std::vector<NodeSocket>& sockets = named.output ? node.outputs : node.inputs;
        const NodeSocketData& data = (named.output ? type.outputs : type.inputs)[named.index];
        if (const auto socket = std::ranges::find(sockets, data.identifier, &NodeSocket::identifier); socket != sockets.end())
            socket->name = named.name;
    }
    std::vector<float> hidden;
    for (std::size_t index = 0; index < node.inputs.size() && index < type.inputs.size(); ++index)
        if (state.hiddenValues >> index & 1)
            hidden.push_back(static_cast<float>(index));
    eraseProperty(node.properties, hideValue);
    if (!hidden.empty())
        node.properties.insert(node.properties.begin(), numbersProperty(hideValue, std::move(hidden)));
}

void updateRamp(Node& node) {
    const std::span<const float> elements = numbers(node.properties, rampElements);
    const std::string interpolation = text(node.properties, "color_ramp.interpolation");
    const std::string colorMode = text(node.properties, "color_ramp.color_mode");
    if (colorMode == "RGB" && interpolation != "B_SPLINE" && interpolation != "CARDINAL") {
        eraseProperty(node.properties, rampSamples);
        return;
    }
    const std::string hue = text(node.properties, "color_ramp.hue_interpolation");
    storeProperty(node.properties, numbersProperty(rampSamples, RampSampler(elements, interpolation, colorMode, hue).sample()));
}

void updateCurves(Node& node, const NodeTypeData& type) {
    const std::string_view kind = nodeType(node.type);
    const auto numbersOf = [&](const std::string& name) {
        if (const NodeProperty* property = findProperty(node.properties, name))
            return std::span<const float>(property->numbers);
        const NodePropertyData* data = findData(type, name);
        return data ? data->numbers : std::span<const float>();
    };
    const bool extrapolates = text(node.properties, "mapping.extend") == "EXTRAPOLATED";
    const std::span<const float> clip = numbersOf("mapping.clip");
    const auto count = static_cast<std::size_t>(std::max<std::int64_t>(integer(node.properties, "mapping.curve_count"), 0));
    std::vector<std::span<const float>> points;
    std::vector<CurveSampler> curves;
    points.reserve(count);
    curves.reserve(count);
    for (std::size_t curve = 0; curve < count; ++curve) {
        points.push_back(numbersOf(std::format("mapping.curve{}", curve)));
        const std::span<const float> stored = numbersOf(std::format("mapping.handles{}", curve));
        std::vector<CurveHandle> handles(points.back().size() / 2, CurveHandle::Auto);
        for (std::size_t index = 0; index < std::min(handles.size(), stored.size()); ++index)
            handles[index] = static_cast<CurveHandle>(stored[index]);
        curves.emplace_back(points.back(), handles, extrapolates, clip[1], clip[3]);
    }
    const auto store = [&](const std::size_t channel, const std::pair<float, float> range, const std::initializer_list<std::size_t> chain) {
        std::vector<float> samples;
        for (const float position : curveSamplePositions(range.first, range.second)) {
            float value = position;
            for (const std::size_t curve : chain)
                value = curves[curve].evaluate(value);
            samples.push_back(value);
        }
        storeProperty(node.properties, numbersProperty(std::format("mapping.samples{}", channel), std::move(samples)));
    };
    if (kind == "CURVE_FLOAT" && count >= 1)
        store(0, curveRange({points[0]}), {0});
    if (kind == "CURVE_RGB" && count >= 4)
        for (std::size_t channel = 0; channel < 3; ++channel)
            store(channel, curveRange({points[3], points[channel]}), {3, channel});
    if (kind == "CURVE_VEC" && count >= 3)
        for (std::size_t channel = 0; channel < 3; ++channel)
            store(channel, curveRange({points[channel]}), {channel});
}

void updateDerived(Node& node, const NodeTypeData& type) {
    if (findProperty(node.properties, rampElements))
        updateRamp(node);
    if (findProperty(node.properties, "mapping.curve_count"))
        updateCurves(node, type);
}

//! The name, or the name with the first free number after it, as Blender numbers names.
std::string uniqueName(const NodeTree& tree, const std::string_view name) {
    std::unordered_set<std::string_view> names;
    for (const Node& node : tree.nodes)
        names.insert(node.name);
    if (!names.contains(name))
        return std::string(name);
    std::string_view base = name;
    std::uint32_t number = 0;
    if (const std::size_t dot = name.rfind('.'); dot != std::string_view::npos && dot + 1 < name.size()) {
        const std::string_view digits = name.substr(dot + 1);
        const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), number);
        if (error == std::errc() && end == digits.data() + digits.size())
            base = name.substr(0, dot);
        else
            number = 0;
    }
    for (;;) {
        std::string candidate = std::format("{}.{:03}", base, ++number);
        if (!names.contains(candidate))
            return candidate;
    }
}

//! Makes one node the active output among the nodes of its type.
void activateOutput(NodeTree& tree, const std::uint32_t active) {
    for (std::uint32_t index = 0; index < tree.nodes.size(); ++index)
        if (tree.nodes[index].type == tree.nodes[active].type)
            if (NodeProperty* property = findMutable(tree.nodes[index].properties, activeOutput))
                property->integer = index == active ? 1 : 0;
}

void repairActiveOutputs(NodeTree& tree) {
    for (std::uint32_t index = 0; index < tree.nodes.size(); ++index) {
        const std::string& type = tree.nodes[index].type;
        if (!findProperty(tree.nodes[index].properties, activeOutput))
            continue;
        const bool active = std::ranges::any_of(tree.nodes, [&](const Node& node) { return node.type == type && flag(node.properties, activeOutput); });
        if (!active)
            activateOutput(tree, index);
    }
}

const NodeLink* findLink(const NodeTree& tree, const std::uint32_t toNode, const std::uint32_t toSocket) {
    const auto found = std::ranges::find_if(tree.links, [&](const NodeLink& link) { return link.toNode == toNode && link.toSocket == toSocket; });
    return found == tree.links.end() ? nullptr : &*found;
}

//! Whether a path of links leads from a node to another.
bool reaches(const NodeTree& tree, const std::uint32_t from, const std::uint32_t to) {
    std::vector<std::vector<std::uint32_t>> next(tree.nodes.size());
    for (const NodeLink& link : tree.links)
        next[link.fromNode].push_back(link.toNode);
    std::vector<bool> seen(tree.nodes.size());
    std::vector<std::uint32_t> pending{from};
    while (!pending.empty()) {
        const std::uint32_t node = pending.back();
        pending.pop_back();
        if (node == to)
            return true;
        if (seen[node])
            continue;
        seen[node] = true;
        pending.insert(pending.end(), next[node].begin(), next[node].end());
    }
    return false;
}

bool isReroute(const NodeTree& tree, const std::uint32_t node) {
    return tree.nodes[node].type == rerouteType;
}

//! The type that a reroute takes: of the output that drives it, else of the first input that it drives; none without either.
std::optional<NodeSocketType> rerouteSocketType(const NodeTree& tree, const std::uint32_t reroute) {
    std::uint32_t top = reroute;
    for (std::size_t step = 0; step < tree.nodes.size(); ++step) {
        const NodeLink* link = findLink(tree, top, 0);
        if (!link)
            break;
        if (!isReroute(tree, link->fromNode))
            return tree.nodes[link->fromNode].outputs[link->fromSocket].type;
        top = link->fromNode;
    }
    std::vector<bool> seen(tree.nodes.size());
    std::vector<std::uint32_t> pending{top};
    for (std::size_t index = 0; index < pending.size(); ++index) {
        if (seen[pending[index]])
            continue;
        seen[pending[index]] = true;
        for (const NodeLink& link : tree.links) {
            if (link.fromNode != pending[index])
                continue;
            if (!isReroute(tree, link.toNode))
                return tree.nodes[link.toNode].inputs[link.toSocket].type;
            pending.push_back(link.toNode);
        }
    }
    return std::nullopt;
}

struct SocketKind {
    NodeSocketType type;
    std::uint8_t components;
    Core::float4 value;
    std::string_view idname;
};

constexpr SocketKind socketKinds[] = {{NodeSocketType::Float, 1, {}, "NodeSocketFloat"},
                                      {NodeSocketType::Int, 1, {}, "NodeSocketInt"},
                                      {NodeSocketType::Bool, 1, {}, "NodeSocketBool"},
                                      {NodeSocketType::Vector, 3, {}, "NodeSocketVector"},
                                      {NodeSocketType::Color, 4, {0.0f, 0.0f, 0.0f, 1.0f}, "NodeSocketColor"},
                                      {NodeSocketType::Shader, 1, {}, "NodeSocketShader"},
                                      {NodeSocketType::String, 1, {}, "NodeSocketString"},
                                      {NodeSocketType::Rotation, 3, {}, "NodeSocketRotation"},
                                      {NodeSocketType::Matrix, 1, {}, "NodeSocketMatrix"},
                                      {NodeSocketType::Menu, 1, {}, "NodeSocketMenu"},
                                      {NodeSocketType::Closure, 1, {}, "NodeSocketClosure"}};

void updateReroutes(NodeTree& tree) {
    for (std::uint32_t index = 0; index < tree.nodes.size(); ++index) {
        if (!isReroute(tree, index))
            continue;
        const std::optional<NodeSocketType> type = rerouteSocketType(tree, index);
        Node& node = tree.nodes[index];
        if (!type || node.inputs.empty() || node.outputs.empty() || node.inputs[0].type == *type)
            continue;
        const auto kind = std::ranges::find(socketKinds, *type, &SocketKind::type);
        if (kind == std::end(socketKinds))
            continue;
        for (NodeSocket* socket : {&node.inputs[0], &node.outputs[0]}) {
            socket->type = kind->type;
            socket->components = kind->components;
            socket->value = kind->value;
        }
        if (NodeProperty* property = findMutable(node.properties, "socket_idname"))
            property->text = kind->idname;
    }
}

//! Drops the groups that no node uses.
void pruneGroups(NodeTree& tree) {
    std::vector<std::uint32_t> remap(tree.groups.size(), noNode);
    for (const Node& node : tree.nodes)
        if (node.group < remap.size())
            remap[node.group] = 0;
    std::vector<std::shared_ptr<const NodeTree>> groups;
    for (std::size_t index = 0; index < remap.size(); ++index) {
        if (remap[index] == noNode)
            continue;
        remap[index] = static_cast<std::uint32_t>(groups.size());
        groups.push_back(std::move(tree.groups[index]));
    }
    tree.groups = std::move(groups);
    for (Node& node : tree.nodes)
        if (node.group < remap.size())
            node.group = remap[node.group];
}

Core::float2 absoluteLocation(const NodeTree& tree, std::uint32_t node) {
    Core::float2 location = tree.nodes[node].location;
    for (std::size_t depth = 0; depth < tree.nodes.size() && tree.nodes[node].parent < tree.nodes.size(); ++depth) {
        node = tree.nodes[node].parent;
        location = location + tree.nodes[node].location;
    }
    return location;
}

//! Sorts the elements of a ramp by position.
void sortElements(std::vector<float>& values) {
    std::vector<std::array<float, rampElementSize>> elements(values.size() / rampElementSize);
    for (std::size_t index = 0; index < values.size(); ++index)
        elements[index / rampElementSize][index % rampElementSize] = values[index];
    std::ranges::stable_sort(elements, {}, [](const auto& element) { return element[0]; });
    for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = elements[index / rampElementSize][index % rampElementSize];
}

//! Sorts the points of a curve by x. The handles follow their points: by index when the count stays, else by place.
void storeCurve(Node& node, const NodeTypeData& type, NodeProperty property) {
    const std::string handlesName = "mapping.handles" + property.name.substr(std::string_view("mapping.curve").size());
    const std::span<const float> oldPoints = numbers(node.properties, property.name);
    const NodePropertyData* defaults = findData(type, handlesName);
    const NodeProperty* stored = findProperty(node.properties, handlesName);
    const std::span<const float> oldHandles = stored ? std::span<const float>(stored->numbers) : defaults ? defaults->numbers : std::span<const float>();
    struct Point {
        float x;
        float y;
        float handle;
    };
    std::vector<Point> points;
    for (std::size_t index = 0; index + 1 < property.numbers.size(); index += 2) {
        Point point{property.numbers[index], property.numbers[index + 1], 0.0f};
        std::size_t old = index / 2;
        if (oldPoints.size() != property.numbers.size()) {
            old = 0;
            while (old * 2 + 1 < oldPoints.size() && (oldPoints[old * 2] != point.x || oldPoints[old * 2 + 1] != point.y))
                ++old;
        }
        if (old < oldHandles.size())
            point.handle = oldHandles[old];
        points.push_back(point);
    }
    std::ranges::stable_sort(points, {}, &Point::x);
    std::vector<float> handles;
    for (std::size_t index = 0; index < points.size(); ++index) {
        property.numbers[index * 2] = points[index].x;
        property.numbers[index * 2 + 1] = points[index].y;
        handles.push_back(points[index].handle);
    }
    storeProperty(node.properties, std::move(property));
    storeProperty(node.properties, numbersProperty(handlesName, std::move(handles)));
}

bool isCurve(const std::string_view name) {
    constexpr std::string_view prefix = "mapping.curve";
    return name.size() == prefix.size() + 1 && name.starts_with(prefix);
}

std::expected<void, Core::Error> validate(const NodePropertyData& setting, const NodeProperty& property) {
    if (property.kind != setting.kind)
        return Core::fail(Core::ErrorCode::InvalidArgument, std::format("the setting '{}' has another kind", property.name));
    if (!setting.items.empty() && !std::ranges::contains(setting.items, std::string_view(property.text), &NodeEnumData::identifier))
        return Core::fail(Core::ErrorCode::InvalidArgument, std::format("the setting '{}' has no choice '{}'", property.name, property.text));
    if (property.name == rampElements && (property.numbers.empty() || property.numbers.size() % rampElementSize != 0))
        return Core::fail(Core::ErrorCode::InvalidArgument, "a colour ramp needs five values for each element");
    if (isCurve(property.name) && (property.numbers.size() < 4 || property.numbers.size() % 2 != 0))
        return Core::fail(Core::ErrorCode::InvalidArgument, "a curve needs two or more points");
    if (property.name == "mapping.clip" && property.numbers.size() != setting.numbers.size())
        return Core::fail(Core::ErrorCode::InvalidArgument, "the clip of a curve needs five values");
    return {};
}

std::string lower(const std::string_view text) {
    std::string result(text);
    for (char& letter : result)
        if (letter >= 'A' && letter <= 'Z')
            letter = static_cast<char>(letter - 'A' + 'a');
    return result;
}

//! How well a name matches a lower case text, 0 best; none without a match.
std::optional<int> matchRank(const std::string& name, const std::string& wanted) {
    if (name == wanted)
        return 0;
    const std::size_t found = name.find(wanted);
    if (found == 0)
        return 1;
    if (name.find(' ' + wanted) != std::string::npos)
        return 2;
    return found == std::string::npos ? std::nullopt : std::optional(3);
}

}

const NodeTypeData* findNodeTypeData(const std::string_view type) {
    const std::span<const NodeTypeData> types = nodeTypeData();
    const auto found = std::ranges::lower_bound(types, type, {}, &NodeTypeData::type);
    return found == types.end() || found->type != type ? nullptr : &*found;
}

std::span<const NodeType> NodeCatalog::getTypes() {
    static const std::vector<NodeType> types = [] {
        std::vector<NodeType> result;
        result.reserve(nodeTypeData().size());
        for (const NodeTypeData& data : nodeTypeData())
            result.push_back(toType(data));
        return result;
    }();
    return types;
}

const NodeType* NodeCatalog::find(const std::string_view type) {
    const NodeTypeData* data = findNodeTypeData(type);
    return data ? &getTypes()[static_cast<std::size_t>(data - nodeTypeData().data())] : nullptr;
}

std::vector<const NodeType*> NodeCatalog::search(const std::string_view text, const NodeTreeKind kind) {
    struct Match {
        int rank;
        const NodeType* type;
    };
    const std::string wanted = lower(text);
    std::vector<Match> matches;
    for (const NodeType& type : getTypes()) {
        if (!std::ranges::contains(type.trees, kind))
            continue;
        if (const std::optional<int> rank = matchRank(lower(type.name), wanted))
            matches.push_back({wanted.empty() ? 0 : *rank, &type});
    }
    std::ranges::sort(matches, {}, [](const Match& match) { return std::pair<int, std::string_view>(match.rank, match.type->name); });
    std::vector<const NodeType*> types;
    types.reserve(matches.size());
    for (const Match& match : matches)
        types.push_back(match.type);
    return types;
}

std::expected<std::uint32_t, Core::Error> NodeCatalog::addNode(NodeTree& tree, const std::string_view type, const Core::float2 location) {
    const NodeTypeData* data = findNodeTypeData(type);
    if (!data)
        return Core::fail(Core::ErrorCode::InvalidArgument, std::format("there is no node type '{}'", type));
    Node node;
    node.type = data->type;
    node.name = uniqueName(tree, data->name);
    node.location = location;
    node.width = data->width;
    for (const NodeSocketData& socket : data->inputs)
        node.inputs.push_back(toSocket(socket));
    for (const NodeSocketData& socket : data->outputs)
        node.outputs.push_back(toSocket(socket));
    std::vector<float> hidden;
    for (std::size_t index = 0; index < data->inputs.size(); ++index)
        if (data->inputs[index].hideValue)
            hidden.push_back(static_cast<float>(index));
    if (!hidden.empty())
        node.properties.push_back(numbersProperty(hideValue, std::move(hidden)));
    for (const NodePropertyData& property : data->properties)
        if (property.stored)
            node.properties.push_back(toProperty(property));
    applyRules(node, *data);
    updateDerived(node, *data);
    const auto index = static_cast<std::uint32_t>(tree.nodes.size());
    tree.nodes.push_back(std::move(node));
    if (findProperty(tree.nodes[index].properties, activeOutput))
        activateOutput(tree, index);
    return index;
}

void NodeCatalog::removeNodes(NodeTree& tree, const std::span<const std::uint32_t> nodes) {
    std::vector<bool> removed(tree.nodes.size());
    for (const std::uint32_t node : nodes)
        if (node < removed.size())
            removed[node] = true;
    std::vector<Core::float2> locations(tree.nodes.size());
    for (std::uint32_t index = 0; index < tree.nodes.size(); ++index)
        locations[index] = absoluteLocation(tree, index);
    std::vector<std::uint32_t> remap(tree.nodes.size(), noNode);
    std::uint32_t kept = 0;
    for (std::uint32_t index = 0; index < tree.nodes.size(); ++index)
        if (!removed[index])
            remap[index] = kept++;
    for (std::uint32_t index = 0; index < tree.nodes.size(); ++index) {
        Node& node = tree.nodes[index];
        if (node.parent >= removed.size())
            continue;
        if (!removed[node.parent]) {
            node.parent = remap[node.parent];
            continue;
        }
        node.location = locations[index];
        node.parent = noNode;
    }
    for (const NodeLink& link : tree.links)
        if (removed[link.fromNode])
            eraseProperty(tree.nodes[link.toNode].properties, internalLinks);
    std::erase_if(tree.links, [&](const NodeLink& link) { return removed[link.fromNode] || removed[link.toNode]; });
    for (NodeLink& link : tree.links) {
        link.fromNode = remap[link.fromNode];
        link.toNode = remap[link.toNode];
    }
    for (std::uint32_t index = 0; index < removed.size(); ++index)
        if (!removed[index] && remap[index] != index)
            tree.nodes[remap[index]] = std::move(tree.nodes[index]);
    tree.nodes.resize(kept);
    pruneGroups(tree);
    repairActiveOutputs(tree);
    updateReroutes(tree);
}

std::vector<std::uint32_t> NodeCatalog::duplicateNodes(NodeTree& tree, const std::span<const std::uint32_t> nodes, const Core::float2 offset) {
    std::vector<std::uint32_t> copies;
    std::vector<std::uint32_t> remap(tree.nodes.size(), noNode);
    for (const std::uint32_t source : nodes) {
        if (source >= remap.size() || remap[source] != noNode)
            continue;
        remap[source] = static_cast<std::uint32_t>(tree.nodes.size());
        copies.push_back(remap[source]);
        Node copy = tree.nodes[source];
        copy.name = uniqueName(tree, copy.name);
        if (NodeProperty* active = findMutable(copy.properties, activeOutput))
            active->integer = 0;
        tree.nodes.push_back(std::move(copy));
    }
    for (const std::uint32_t index : copies) {
        Node& copy = tree.nodes[index];
        if (copy.parent < remap.size() && remap[copy.parent] != noNode)
            copy.parent = remap[copy.parent];
        else
            copy.location = copy.location + offset;
    }
    for (std::size_t index = 0, count = tree.links.size(); index < count; ++index) {
        NodeLink link = tree.links[index];
        if (remap[link.fromNode] == noNode || remap[link.toNode] == noNode)
            continue;
        link.fromNode = remap[link.fromNode];
        link.toNode = remap[link.toNode];
        tree.links.push_back(link);
    }
    for (const std::uint32_t index : copies)
        if (findNodeTypeData(tree.nodes[index].type))
            eraseProperty(tree.nodes[index].properties, internalLinks);
    updateReroutes(tree);
    return copies;
}

std::expected<void, Core::Error> NodeCatalog::connect(NodeTree& tree, const NodeLink link) {
    if (link.fromNode >= tree.nodes.size() || link.toNode >= tree.nodes.size() || link.fromSocket >= tree.nodes[link.fromNode].outputs.size()
        || link.toSocket >= tree.nodes[link.toNode].inputs.size())
        return Core::fail(Core::ErrorCode::InvalidArgument, "the link has no such socket");
    const NodeSocket& from = tree.nodes[link.fromNode].outputs[link.fromSocket];
    const NodeSocket& to = tree.nodes[link.toNode].inputs[link.toSocket];
    if (from.type == NodeSocketType::Custom || to.type == NodeSocketType::Custom)
        return Core::fail(Core::ErrorCode::InvalidArgument, "a link cannot add a socket");
    if (from.type == NodeSocketType::Shader && to.type != NodeSocketType::Shader && !isReroute(tree, link.toNode))
        return Core::fail(Core::ErrorCode::InvalidArgument, "a shader output only links to a shader input");
    if (reaches(tree, link.toNode, link.fromNode))
        return Core::fail(Core::ErrorCode::InvalidArgument, "the link makes a cycle");
    disconnect(tree, link.toNode, link.toSocket);
    tree.links.push_back(link);
    updateReroutes(tree);
    return {};
}

void NodeCatalog::disconnect(NodeTree& tree, const std::uint32_t toNode, const std::uint32_t toSocket) {
    if (toNode >= tree.nodes.size())
        return;
    std::erase_if(tree.links, [&](const NodeLink& link) { return link.toNode == toNode && link.toSocket == toSocket; });
    if (findNodeTypeData(tree.nodes[toNode].type))
        eraseProperty(tree.nodes[toNode].properties, internalLinks);
    updateReroutes(tree);
}

std::expected<void, Core::Error> NodeCatalog::setProperty(NodeTree& tree, const std::uint32_t node, NodeProperty property) {
    if (node >= tree.nodes.size())
        return Core::fail(Core::ErrorCode::InvalidArgument, "the node does not exist");
    Node& target = tree.nodes[node];
    const NodeTypeData* type = findNodeTypeData(target.type);
    const NodePropertyData* setting = type ? findData(*type, property.name) : nullptr;
    const NodeProperty* existing = findProperty(target.properties, property.name);
    if (!setting && (!existing || existing->kind != property.kind))
        return Core::fail(Core::ErrorCode::InvalidArgument, std::format("the node '{}' has no setting '{}'", target.name, property.name));
    if (!setting) {
        storeProperty(target.properties, std::move(property));
        return {};
    }
    if (const std::expected<void, Core::Error> valid = validate(*setting, property); !valid)
        return valid;
    if (setting->minimum <= setting->maximum) {
        if (property.kind == NodePropertyKind::Number)
            property.number.x = std::clamp(property.number.x, setting->minimum, setting->maximum);
        else if (property.kind == NodePropertyKind::Integer)
            property.integer = std::clamp(property.integer, static_cast<std::int64_t>(setting->minimum), static_cast<std::int64_t>(setting->maximum));
    }
    if (property.name == rampElements)
        sortElements(property.numbers);
    const bool activates = property.name == activeOutput && property.integer != 0;
    if (isCurve(property.name))
        storeCurve(target, *type, std::move(property));
    else
        storeProperty(target.properties, std::move(property));
    if (activates)
        activateOutput(tree, node);
    eraseProperty(target.properties, internalLinks);
    applyRules(target, *type);
    updateDerived(target, *type);
    return {};
}

std::expected<void, Core::Error> NodeCatalog::renameNode(NodeTree& tree, const std::uint32_t node, std::string name) {
    if (node >= tree.nodes.size())
        return Core::fail(Core::ErrorCode::InvalidArgument, "the node does not exist");
    if (name.empty())
        return Core::fail(Core::ErrorCode::InvalidArgument, "a node needs a name");
    if (std::ranges::any_of(tree.nodes, [&](const Node& other) { return &other != &tree.nodes[node] && other.name == name; }))
        return Core::fail(Core::ErrorCode::InvalidArgument, std::format("the tree already has a node named {}", name));
    tree.nodes[node].name = std::move(name);
    return {};
}

std::expected<void, Core::Error> NodeCatalog::setTexture(NodeTree& tree, const std::uint32_t node, const Scene::Scene& scene,
                                                         const Scene::TextureHandle texture) {
    if (node >= tree.nodes.size())
        return Core::fail(Core::ErrorCode::InvalidArgument, "the node does not exist");
    Node& target = tree.nodes[node];
    const std::string_view kind = nodeType(target.type);
    if (kind != "TEX_IMAGE" && kind != "TEX_ENVIRONMENT")
        return Core::fail(Core::ErrorCode::InvalidArgument, std::format("the node '{}' takes no image", target.name));
    const Scene::Texture* image = scene.find(texture);
    if (texture && !image)
        return Core::fail(Core::ErrorCode::InvalidArgument, "the texture does not exist");
    std::erase_if(target.properties, isTextureRecord);
    eraseProperty(target.properties, "image");
    eraseProperty(target.properties, "image_source");
    if (!image)
        return {};
    setNodeImage(tree, node, image->name, texture);
    const std::vector<Core::float4> samples = sampleImage(*image);
    if (!samples.empty())
        addTextureRecord(target, samplesRecord(samples));
    return {};
}

void NodeCatalog::addDefaultNodes(NodeTree& tree, const NodeTreeKind kind) {
    const NodeTreeDefault& data = *std::ranges::find(nodeTreeDefaults(), kind, &NodeTreeDefault::kind);
    if (tree.name.empty())
        tree.name = data.name;
    for (const NodePropertyData& property : data.properties)
        if (!findProperty(tree.properties, property.name))
            tree.properties.push_back(toProperty(property));
    const auto base = static_cast<std::uint32_t>(tree.nodes.size());
    for (const NodeDefault& node : data.nodes)
        if (const std::expected<std::uint32_t, Core::Error> added = addNode(tree, node.type, node.location))
            tree.nodes[*added].width = node.width;
    for (NodeLink link : data.links) {
        link.fromNode += base;
        link.toNode += base;
        tree.links.push_back(link);
    }
    for (const NodeDefaultValue& value : data.values)
        tree.nodes[base + value.node].inputs[value.input].value = value.value;
}

void NodeCatalog::setLightOutput(NodeTree& tree, const Scene::Light& compiled, const float power, const Core::float3 color) {
    const float energy = number(tree.properties, "energy");
    if (power != compiled.power) {
        //! Without a power to scale, the shader strength and the area factor count as one.
        const float scaled = compiled.power > 0.0f && energy != 0.0f ? energy * power / compiled.power : power / std::exp2(number(tree.properties, "exposure"));
        NodeProperty property;
        property.name = "energy";
        property.number.x = scaled;
        storeProperty(tree.properties, std::move(property));
    }
    const std::array<float, 3> wanted{color.x, color.y, color.z};
    const std::array<float, 3> current{compiled.color.x, compiled.color.y, compiled.color.z};
    if (wanted == current)
        return;
    const NodeValues base = values(tree.properties, "color");
    std::vector<float> next(3);
    for (std::size_t channel = 0; channel < next.size(); ++channel) {
        const float setting = base.count == 0 ? 1.0f : base.values[std::min(channel, base.count - 1)];
        next[channel] = current[channel] > 0.0f && setting > 0.0f ? setting * wanted[channel] / current[channel] : wanted[channel];
    }
    NodeProperty* existing = findMutable(tree.properties, "color");
    if (existing && existing->kind != NodePropertyKind::Numbers) {
        existing->number = {next[0], next[1], next[2], existing->number.w};
        return;
    }
    storeProperty(tree.properties, numbersProperty("color", std::move(next)));
}

std::expected<void, Core::Error> NodeCatalog::setEnvironmentOutput(NodeTree& tree, const Scene::Environment& compiled, const Scene::Environment& wanted) {
    const bool strength = wanted.strength != compiled.strength;
    const bool color = wanted.color != compiled.color;
    const bool rotation = wanted.rotation != compiled.rotation;
    NodeLinks links;
    const NodeRef output = flag(tree.properties, "use_nodes") ? findOutput(tree, "ShaderNodeOutputWorld") : NodeRef();
    const NodeLinkEnd surface = output ? links.linkedNode(inputNamed(output, "Surface")) : NodeLinkEnd{};
    const bool background = surface.node && nodeType(surface.node) == "BACKGROUND" && !surface.node->muted && !compiled.graph;
    const SocketRef strengthSocket = background ? inputNamed(surface.node, "Strength") : SocketRef();
    const SocketRef colorSocket = background ? inputNamed(surface.node, "Color") : SocketRef();
    if ((strength || color || rotation) && !background)
        return Core::fail(Core::ErrorCode::Unsupported, "the world shader is not a plain background; change it in the node editor");
    if (strength && (!strengthSocket || links.isLinked(strengthSocket)))
        return Core::fail(Core::ErrorCode::Unsupported, "the background strength is linked; change it in the node editor");
    if (color && (!colorSocket || links.isLinked(colorSocket)))
        return Core::fail(Core::ErrorCode::Unsupported, "the background colour is linked; change it in the node editor");
    if (strength)
        tree.nodes[surface.node.index].inputs[strengthSocket.index].value.x = wanted.strength;
    if (color) {
        Core::float4& value = tree.nodes[surface.node.index].inputs[colorSocket.index].value;
        value = {wanted.color.x, wanted.color.y, wanted.color.z, value.w};
    }
    if (rotation) {
        NodeProperty property;
        property.name = "noorray.rotation";
        property.number.x = number(tree.properties, "noorray.rotation") + wanted.rotation - compiled.rotation;
        storeProperty(tree.properties, std::move(property));
    }
    if (wanted.visibleToCamera == compiled.visibleToCamera)
        return {};
    NodeProperty camera;
    camera.name = "cycles_visibility.camera";
    camera.kind = NodePropertyKind::Boolean;
    camera.integer = wanted.visibleToCamera ? 1 : 0;
    storeProperty(tree.properties, std::move(camera));
    if (!wanted.visibleToCamera || !flag(tree.properties, "film_transparent"))
        return {};
    NodeProperty film;
    film.name = "film_transparent";
    film.kind = NodePropertyKind::Boolean;
    storeProperty(tree.properties, std::move(film));
    return {};
}

}
