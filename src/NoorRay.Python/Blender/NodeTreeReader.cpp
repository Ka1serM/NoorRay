#include "NoorRay.Python/Blender/NodeTreeReader.h"

#include <algorithm>
#include <array>
#include <format>
#include <string_view>

#include "NoorRay/Materials/Nodes/NodeAccess.h"
#include "NoorRay/Materials/Nodes/NodeTextures.h"

namespace NoorRay::Python {

namespace {

using Materials::NodeProperty;
using Materials::NodePropertyKind;
using Materials::NodeSocketType;

NodeSocketType toSocketType(const std::string_view type) {
    constexpr std::pair<std::string_view, NodeSocketType> types[] = {
        {"VALUE", NodeSocketType::Float},   {"INT", NodeSocketType::Int},       {"BOOLEAN", NodeSocketType::Bool},    {"VECTOR", NodeSocketType::Vector},
        {"RGBA", NodeSocketType::Color},    {"SHADER", NodeSocketType::Shader}, {"STRING", NodeSocketType::String},   {"ROTATION", NodeSocketType::Rotation},
        {"MATRIX", NodeSocketType::Matrix}, {"MENU", NodeSocketType::Menu},     {"CLOSURE", NodeSocketType::Closure},
    };
    const auto found = std::ranges::find(types, type, &std::pair<std::string_view, NodeSocketType>::first);
    return found == std::end(types) ? NodeSocketType::Custom : found->second;
}

bool hasValue(const NodeSocketType type) {
    return type == NodeSocketType::Float || type == NodeSocketType::Int || type == NodeSocketType::Bool || type == NodeSocketType::Vector
           || type == NodeSocketType::Color || type == NodeSocketType::Rotation;
}

//! The nodes whose image, profile or sky the texture source loads from Blender.
bool holdsTexture(const std::string_view idname) {
    return idname == "ShaderNodeTexImage" || idname == "ShaderNodeTexEnvironment" || idname == "ShaderNodeTexIES" || idname == "ShaderNodeTexSky";
}

//! The handle type of a curve point, by its place in Blender's enum.
float handleIndex(const std::string& type) {
    if (type == "AUTO_CLAMPED")
        return 1.0f;
    return type == "VECTOR" ? 2.0f : 0.0f;
}

std::uint32_t indexOf(const std::vector<Materials::NodeSocket>& sockets, const std::string& identifier) {
    for (std::uint32_t index = 0; index < sockets.size(); ++index)
        if (sockets[index].identifier == identifier)
            return index;
    return Materials::noNode;
}

}

Object nodeTree(Reader& reader, const Object& owner) {
    const Object tree = reader.find(owner, "node_tree");
    if (!tree)
        return {};
    const float major = reader.values(reader.at(reader.get(reader.get(reader.module("bpy"), "app"), "version"), 0)).values[0];
    if (major >= 5.0f || !reader.find(owner, "use_nodes") || reader.flag(owner, "use_nodes"))
        return tree;
    return {};
}

NodeProperty booleanProperty(std::string name, const bool value) {
    return {.name = std::move(name), .kind = NodePropertyKind::Boolean, .integer = value ? 1 : 0};
}

NodeProperty integerProperty(std::string name, const std::int64_t value) {
    return {.name = std::move(name), .kind = NodePropertyKind::Integer, .integer = value};
}

NodeProperty numberProperty(std::string name, const float value) {
    return {.name = std::move(name), .kind = NodePropertyKind::Number, .number = {value, 0.0f, 0.0f, 0.0f}};
}

NodeProperty textProperty(std::string name, std::string value) {
    return {.name = std::move(name), .kind = NodePropertyKind::Text, .text = std::move(value)};
}

NodeProperty numbersProperty(std::string name, const std::span<const float> values) {
    return {.name = std::move(name), .kind = NodePropertyKind::Numbers, .numbers = {values.begin(), values.end()}};
}

void NodeTreeReader::addRecord(const Materials::NodeTree& tree, const std::uint32_t node, Materials::NodeProperty record) {
    const auto owns = [&](const std::shared_ptr<Materials::NodeTree>& owned) { return owned.get() == &tree; };
    if (const auto root = std::ranges::find_if(roots, owns); root != roots.end())
        return Materials::addTextureRecord((*root)->nodes[node], std::move(record));
    if (const auto group = std::ranges::find_if(groups, [&](const auto& entry) { return owns(entry.second); }); group != groups.end())
        Materials::addTextureRecord(group->second->nodes[node], std::move(record));
}

std::shared_ptr<Materials::NodeTree> NodeTreeReader::read(const Object& tree) {
    auto result = std::make_shared<Materials::NodeTree>();
    roots.push_back(result);
    if (tree)
        readTree(tree, *result);
    return result;
}

Object NodeTreeReader::find(const Materials::NodeTree& tree, const std::uint32_t node) const {
    const auto found = nodes.find({&tree, node});
    return found == nodes.end() ? Object() : found->second;
}

void NodeTreeReader::readTree(const Object& source, Materials::NodeTree& tree) {
    tree.name = reader.text(source, "name");
    const std::vector<Object> sources = reader.items(reader.get(source, "nodes"));
    tree.nodes.resize(sources.size());
    std::map<std::string, std::uint32_t> names;
    std::vector<std::string> parents(sources.size());
    for (std::uint32_t index = 0; index < sources.size(); ++index) {
        tree.nodes[index].name = reader.text(sources[index], "name");
        names.emplace(tree.nodes[index].name, index);
        if (const Object parent = reader.find(sources[index], "parent"))
            parents[index] = reader.text(parent, "name");
    }
    std::map<std::uintptr_t, std::uint32_t> groupIndices;
    for (std::uint32_t index = 0; index < sources.size(); ++index) {
        readNode(sources[index], tree, index, groupIndices);
        if (const auto parent = names.find(parents[index]); !parents[index].empty() && parent != names.end())
            tree.nodes[index].parent = parent->second;
    }
    for (const Object& link : reader.items(reader.get(source, "links"))) {
        const auto from = names.find(reader.text(reader.get(link, "from_node"), "name"));
        const auto to = names.find(reader.text(reader.get(link, "to_node"), "name"));
        if (from == names.end() || to == names.end())
            continue;
        const std::uint32_t fromSocket = indexOf(tree.nodes[from->second].outputs, reader.text(reader.get(link, "from_socket"), "identifier"));
        const std::uint32_t toSocket = indexOf(tree.nodes[to->second].inputs, reader.text(reader.get(link, "to_socket"), "identifier"));
        if (fromSocket == Materials::noNode || toSocket == Materials::noNode)
            continue;
        tree.links.push_back({from->second, fromSocket, to->second, toSocket, reader.flag(link, "is_muted")});
    }
}

void NodeTreeReader::readNode(const Object& source, Materials::NodeTree& tree, const std::uint32_t index,
                              std::map<std::uintptr_t, std::uint32_t>& groupIndices) {
    Materials::Node& node = tree.nodes[index];
    node.type = reader.text(source, "bl_idname");
    node.label = reader.text(source, "label");
    const Values location = reader.values(source, "location");
    node.location = {location.values[0], location.values[1]};
    node.width = reader.number(source, "width");
    node.muted = reader.flag(source, "mute");
    node.hidden = reader.flag(source, "hide");
    const Object inputs = reader.get(source, "inputs");
    readSockets(inputs, node.inputs);
    readSockets(reader.get(source, "outputs"), node.outputs);
    std::vector<float> hidden;
    for (std::size_t socket = 0; const Object& input : reader.items(inputs)) {
        if (reader.flag(input, "hide_value"))
            hidden.push_back(static_cast<float>(socket));
        ++socket;
    }
    if (!hidden.empty())
        node.properties.push_back(numbersProperty("hide_value", hidden));
    if (node.muted) {
        std::vector<float> pairs;
        for (const Object& link : reader.items(reader.get(source, "internal_links"))) {
            const std::uint32_t from = indexOf(node.inputs, reader.text(reader.get(link, "from_socket"), "identifier"));
            const std::uint32_t to = indexOf(node.outputs, reader.text(reader.get(link, "to_socket"), "identifier"));
            if (from != Materials::noNode && to != Materials::noNode)
                pairs.insert(pairs.end(), {static_cast<float>(from), static_cast<float>(to)});
        }
        node.properties.push_back(numbersProperty("internal_links", pairs));
    }
    for (const Setting& setting : settings(source, node.type))
        readSetting(source, setting, node);
    if (Materials::nodeType(node.type) == "AMBIENT_OCCLUSION")
        node.properties.push_back(numberProperty("world_distance", worldDistance()));
    if (const Object group = Materials::nodeType(node.type) == "GROUP" ? reader.find(source, "node_tree") : Object()) {
        const std::uintptr_t pointer = reader.pointer(group);
        auto shared = groups.find(pointer);
        if (shared == groups.end()) {
            auto inner = std::make_shared<Materials::NodeTree>();
            shared = groups.emplace(pointer, inner).first;
            readTree(group, *inner);
        }
        const auto [found, added] = groupIndices.try_emplace(pointer, static_cast<std::uint32_t>(tree.groups.size()));
        if (added)
            tree.groups.push_back(shared->second);
        node.group = found->second;
    }
    if (holdsTexture(node.type))
        nodes.emplace(std::pair{&tree, index}, source);
}

void NodeTreeReader::readSockets(const Object& collection, std::vector<Materials::NodeSocket>& sockets) {
    for (const Object& source : reader.items(collection)) {
        Materials::NodeSocket& socket = sockets.emplace_back();
        socket.identifier = reader.text(source, "identifier");
        socket.name = reader.text(source, "name");
        socket.type = toSocketType(reader.text(source, "type"));
        socket.enabled = reader.flag(source, "enabled");
        socket.hidden = reader.flag(source, "hide");
        const Object value = reader.find(source, "default_value");
        if (!value)
            continue;
        if (socket.type == NodeSocketType::String || socket.type == NodeSocketType::Menu) {
            socket.text = reader.text(value);
            continue;
        }
        if (!hasValue(socket.type))
            continue;
        const Values values = reader.values(value);
        socket.components = static_cast<std::uint32_t>(values.count);
        socket.value = {values.values[0], values.values[1], values.values[2], values.values[3]};
    }
}

const std::vector<NodeTreeReader::Setting>& NodeTreeReader::settings(const Object& node, const std::string& idname) {
    if (const auto found = types.find(idname); found != types.end())
        return found->second;
    if (baseSettings.empty())
        for (const Object& property :
             reader.items(reader.get(reader.get(reader.get(reader.get(reader.module("bpy"), "types"), "Node"), "bl_rna"), "properties")))
            baseSettings.push_back(reader.text(property, "identifier"));
    std::vector<Setting>& result = types[idname];
    for (const Object& property : reader.items(reader.get(reader.get(node, "bl_rna"), "properties"))) {
        Setting setting{.identifier = reader.text(property, "identifier"), .type = reader.text(property, "type"), .length = 0};
        if (std::ranges::contains(baseSettings, setting.identifier))
            continue;
        if (setting.type == "ENUM" && reader.flag(property, "is_enum_flag"))
            continue;
        if (setting.type == "BOOLEAN" || setting.type == "INT" || setting.type == "FLOAT")
            setting.length = static_cast<std::size_t>(reader.integer(property, "array_length"));
        if (setting.length > 4)
            continue;
        result.push_back(std::move(setting));
    }
    return result;
}

void NodeTreeReader::readSetting(const Object& source, const Setting& setting, Materials::Node& node) {
    const std::string& name = setting.identifier;
    const Object value = reader.attribute(source, name);
    if (setting.type == "POINTER") {
        if (!value)
            return;
        if (name == "color_ramp") {
            readRamp(value, node);
        } else if (name == "mapping") {
            readCurves(value, node);
        } else if (name == "image") {
            node.properties.push_back(textProperty(name, reader.text(value, "name")));
            node.properties.push_back(textProperty("image_source", reader.text(value, "source")));
        } else if (name == "object") {
            node.properties.push_back(textProperty(name, reader.text(value, "name")));
            const Core::Transform inverse = reader.matrix(reader.call(reader.get(value, "matrix_world"), "inverted"));
            std::array<float, 12> rows{};
            for (std::size_t row = 0; row < 3; ++row)
                for (std::size_t column = 0; column < 4; ++column)
                    rows[row * 4 + column] = inverse.rows[row][column];
            node.properties.push_back(numbersProperty("object_inverse", rows));
        } else if (name == "ies") {
            node.properties.push_back(textProperty(name, reader.text(value, "name")));
        }
        return;
    }
    if (setting.type == "COLLECTION") {
        node.properties.push_back(integerProperty(name, static_cast<std::int64_t>(reader.size(value))));
        return;
    }
    if (!value)
        return;
    if (setting.type == "ENUM" || setting.type == "STRING") {
        node.properties.push_back(textProperty(name, reader.text(value)));
        return;
    }
    const Values values = reader.values(value);
    if (setting.length > 0) {
        node.properties.push_back(numbersProperty(name, std::span(values.values).first(values.count)));
        return;
    }
    if (setting.type == "BOOLEAN")
        node.properties.push_back(booleanProperty(name, values.values[0] != 0.0f));
    else if (setting.type == "INT")
        node.properties.push_back(integerProperty(name, static_cast<std::int64_t>(values.values[0])));
    else if (setting.type == "FLOAT")
        node.properties.push_back(numberProperty(name, values.values[0]));
}

void NodeTreeReader::readRamp(const Object& ramp, Materials::Node& node) {
    const std::string interpolation = reader.text(ramp, "interpolation");
    const std::string colorMode = reader.text(ramp, "color_mode");
    node.properties.push_back(textProperty("color_ramp.interpolation", interpolation));
    node.properties.push_back(textProperty("color_ramp.color_mode", colorMode));
    node.properties.push_back(textProperty("color_ramp.hue_interpolation", reader.text(ramp, "hue_interpolation")));
    std::vector<float> elements;
    for (const Object& element : reader.items(reader.get(ramp, "elements"))) {
        const Values color = reader.values(element, "color");
        elements.insert(elements.end(), {reader.number(element, "position"), color.values[0], color.values[1], color.values[2], color.values[3]});
    }
    node.properties.push_back(numbersProperty("color_ramp.elements", elements));
    if (colorMode == "RGB" && interpolation != "B_SPLINE" && interpolation != "CARDINAL")
        return;
    const Materials::RampPieces pieces = Materials::rampPieces(elements);
    std::vector<float> samples;
    const auto evaluate = [&](const float position) {
        const std::array arguments{Object(PyFloat_FromDouble(position))};
        const std::vector<Object> components = reader.items(reader.call(ramp, "evaluate", arguments));
        std::array<float, 4> color{};
        for (std::size_t index = 0; index < components.size() && index < color.size(); ++index)
            color[index] = static_cast<float>(PyFloat_AsDouble(components[index].get()));
        samples.insert(samples.end(), color.begin(), color.end());
    };
    evaluate(pieces.first);
    for (const Materials::RampPiece& piece : pieces.pieces)
        evaluate(piece.start + piece.step);
    node.properties.push_back(numbersProperty("color_ramp.samples", samples));
}

void NodeTreeReader::readCurves(const Object& mapping, Materials::Node& node) {
    reader.call(mapping, "initialize");
    const std::vector<Object> curves = reader.items(reader.get(mapping, "curves"));
    node.properties.push_back(integerProperty("mapping.curve_count", static_cast<std::int64_t>(curves.size())));
    node.properties.push_back(textProperty("mapping.extend", reader.text(mapping, "extend")));
    const std::array clip{reader.number(mapping, "use_clip"), reader.number(mapping, "clip_min_x"), reader.number(mapping, "clip_min_y"),
                          reader.number(mapping, "clip_max_x"), reader.number(mapping, "clip_max_y")};
    node.properties.push_back(numbersProperty("mapping.clip", clip));
    std::vector<std::vector<float>> points(curves.size());
    for (std::size_t curve = 0; curve < curves.size(); ++curve) {
        std::vector<float> handles;
        for (const Object& point : reader.items(reader.get(curves[curve], "points"))) {
            const Values location = reader.values(point, "location");
            points[curve].insert(points[curve].end(), {location.values[0], location.values[1]});
            handles.push_back(handleIndex(reader.text(point, "handle_type")));
        }
        node.properties.push_back(numbersProperty(std::format("mapping.curve{}", curve), points[curve]));
        node.properties.push_back(numbersProperty(std::format("mapping.handles{}", curve), handles));
    }
    const auto sample = [&](const std::size_t channel, std::initializer_list<std::size_t> chain) {
        std::vector<std::span<const float>> used;
        for (const std::size_t curve : chain)
            used.emplace_back(points[curve]);
        const auto [low, high] = used.size() == 1 ? Materials::curveRange({used[0]}) : Materials::curveRange({used[0], used[1]});
        std::vector<float> values;
        for (const float position : Materials::curveSamplePositions(low, high)) {
            float y = position;
            for (const std::size_t curve : chain) {
                const std::array arguments{curves[curve], Object(PyFloat_FromDouble(y))};
                y = static_cast<float>(PyFloat_AsDouble(reader.call(mapping, "evaluate", arguments).get()));
            }
            values.push_back(y);
        }
        node.properties.push_back(numbersProperty(std::format("mapping.samples{}", channel), values));
    };
    const std::string_view kind = Materials::nodeType(node.type);
    if (kind == "CURVE_FLOAT" && !curves.empty())
        sample(0, {0});
    if (kind == "CURVE_RGB" && curves.size() >= 4)
        for (std::size_t channel = 0; channel < 3; ++channel)
            sample(channel, {3, channel});
    if (kind == "CURVE_VEC" && curves.size() >= 3)
        for (std::size_t channel = 0; channel < 3; ++channel)
            sample(channel, {channel});
}

//! Cycles' AO distance of the world, which an AO node with an unlinked zero distance takes.
float NodeTreeReader::worldDistance() {
    if (aoDistance)
        return *aoDistance;
    const Object world = reader.find(reader.get(reader.get(reader.module("bpy"), "context"), "scene"), "world");
    aoDistance = world ? reader.number(reader.get(world, "light_settings"), "distance") : 10.0f;
    return *aoDistance;
}

}
