#include "NoorRay/Materials/Nodes/GraphCompiler.h"

#include <algorithm>
#include <cmath>
#include <expected>
#include <format>
#include <numbers>
#include <span>

#include "NoorRay/Materials/Nodes/NodeCatalogData.h"

namespace NoorRay::Materials {

namespace {

using Materials::GraphFunction;
using Materials::GraphInput;
using Materials::GraphOp;
using Materials::GraphOutput;
using Value = std::uint32_t;

constexpr Core::float3 luminanceWeights{0.2126f, 0.7152f, 0.0722f};
constexpr Core::float3 averageWeights{1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};

std::uint8_t widthOf(const std::string_view type) {
    return type == "VECTOR" || type == "RGBA" ? 3 : 1;
}

bool isValueType(const std::string_view type) {
    return type == "VALUE" || type == "INT" || type == "BOOLEAN" || type == "VECTOR" || type == "RGBA";
}

template <std::size_t Size>
float indexOf(const std::array<std::string_view, Size>& names, const std::string_view name) {
    const auto found = std::ranges::find(names, name);
    return found == names.end() ? 0.0f : static_cast<float>(found - names.begin());
}

std::optional<Scene::TextureWrap> toWrap(const std::string_view name) {
    if (name == "REPEAT")
        return Scene::TextureWrap::Repeat;
    if (name == "EXTEND")
        return Scene::TextureWrap::Extend;
    if (name == "CLIP")
        return Scene::TextureWrap::Clip;
    if (name == "MIRROR")
        return Scene::TextureWrap::Mirror;
    return std::nullopt;
}

Scene::TextureFilter toFilter(const std::string_view name) {
    if (name == "Closest")
        return Scene::TextureFilter::Closest;
    if (name == "Cubic" || name == "Smart")
        return Scene::TextureFilter::Cubic;
    return Scene::TextureFilter::Linear;
}

std::uintptr_t address(const NodeRef node) {
    return reinterpret_cast<std::uintptr_t>(&node.get());
}

//! Short names for the graph operations of one builder.
struct Ops {
    Materials::GraphBuilder& b;

    Value c(const float value) { return b.constant(value); }
    Value c(const Core::float3 value) { return b.constant(value); }
    Value c(const Core::float4 value) { return b.constant(value); }
    Value add(const Value x, const Value y) { return b.op(GraphOp::Add, x, y); }
    Value sub(const Value x, const Value y) { return b.op(GraphOp::Sub, x, y); }
    Value mul(const Value x, const Value y) { return b.op(GraphOp::Mul, x, y); }
    Value div(const Value x, const Value y) { return b.op(GraphOp::Div, x, y); }
    Value mulAdd(const Value x, const Value y, const Value z) { return b.op(GraphOp::MulAdd, x, y, z); }
    Value neg(const Value x) { return sub(c(0.0f), x); }
    Value min(const Value x, const Value y) { return b.op(GraphOp::Min, x, y); }
    Value max(const Value x, const Value y) { return b.op(GraphOp::Max, x, y); }
    Value abs(const Value x) { return b.op(GraphOp::Abs, x); }
    Value floor(const Value x) { return b.op(GraphOp::Floor, x); }
    Value sin(const Value x) { return b.op(GraphOp::Sin, x); }
    Value cos(const Value x) { return b.op(GraphOp::Cos, x); }
    Value less(const Value x, const Value y) { return b.op(GraphOp::Less, x, y); }
    Value greater(const Value x, const Value y) { return b.op(GraphOp::Greater, x, y); }
    Value select(const Value condition, const Value x, const Value y) { return b.op(GraphOp::Select, condition, x, y); }
    Value mix(const Value x, const Value y, const Value t) { return b.op(GraphOp::Mix, x, y, t); }
    Value saturate(const Value x) { return b.op(GraphOp::Clamp, x, c(0.0f), c(1.0f)); }
    Value dot(const Value x, const Value y) { return b.op(GraphOp::Dot, x, y); }
    Value cross(const Value x, const Value y) { return b.op(GraphOp::Cross, x, y); }
    Value normalize(const Value x) { return b.op(GraphOp::Normalize, x); }
    Value x(const Value v, const std::uint32_t component) { return b.extract(v, component); }
    Value combine(std::initializer_list<Value> parts) { return b.combine(std::span(parts.begin(), parts.size())); }
    Value rgb(const Value v) { return b.resize(v, 3); }
    Value call(const GraphFunction function, std::initializer_list<Value> operands) { return b.call(function, std::span(operands.begin(), operands.size())); }
    //! Blender's wrapf: value wrapped into [low, high), or low for an empty range.
    Value wrap(const Value value, const Value high, const Value low) {
        const Value range = sub(high, low);
        const Value wrapped = sub(value, mul(range, floor(div(sub(value, low), range))));
        return select(greater(abs(range), c(0.0f)), wrapped, low);
    }
    //! NoorRay's world is y-up and Blender's z-up: a graph sees a world vector v as Blender does, (v.x, -v.z, v.y), and toNoorRay turns it back.
    Value toBlender(const Value v) {
        const Value turned = combine({x(v, 0), neg(x(v, 2)), x(v, 1)});
        return b.width(v) == 4 ? combine({turned, x(v, 3)}) : turned;
    }
    Value toNoorRay(const Value v) { return combine({x(v, 0), x(v, 2), neg(x(v, 1))}); }
    //! The point or direction p through the rows of a 3x4 transform.
    Value transform(const std::array<Value, 3>& rows, const Value p, const bool point) {
        std::array<Value, 3> parts{};
        for (std::size_t row = 0; row < 3; ++row) {
            parts[row] = dot(rgb(rows[row]), p);
            if (point)
                parts[row] = add(parts[row], x(rows[row], 3));
        }
        return combine({parts[0], parts[1], parts[2]});
    }
    //! The direction p through the transpose of the 3x3 of the rows, which turns a normal by the inverse of the transform.
    Value transposed(const std::array<Value, 3>& rows, const Value p) {
        return mulAdd(rgb(rows[0]), x(p, 0), mulAdd(rgb(rows[1]), x(p, 1), mul(rgb(rows[2]), x(p, 2))));
    }
    //! The direction p through the transpose of the inverse of the 3x3 of the rows: the dot products with the columns of the inverse.
    Value inverseTransposed(const std::array<Value, 3>& rows, const Value p) {
        const Value r0 = rgb(rows[0]);
        const Value r1 = rgb(rows[1]);
        const Value r2 = rgb(rows[2]);
        const Value c0 = cross(r1, r2);
        return div(combine({dot(c0, p), dot(cross(r2, r0), p), dot(cross(r0, r1), p)}), dot(r0, c0));
    }
    //! The point or direction p through the inverse of a 3x4 transform: the inverse 3x3 columns are the cross products of the rows over the determinant.
    Value inverseTransform(const std::array<Value, 3>& rows, const Value p, const bool point) {
        const Value r0 = rgb(rows[0]);
        const Value r1 = rgb(rows[1]);
        const Value r2 = rgb(rows[2]);
        const Value c0 = cross(r1, r2);
        const Value v = point ? sub(p, combine({x(rows[0], 3), x(rows[1], 3), x(rows[2], 3)})) : p;
        const Value turned = mulAdd(c0, x(v, 0), mulAdd(cross(r2, r0), x(v, 1), mul(cross(r0, r1), x(v, 2))));
        return div(turned, dot(r0, c0));
    }
    Value smoothMin(const Value x, const Value y, const Value k) {
        const Value h = div(max(sub(k, abs(sub(x, y))), c(0.0f)), k);
        return sub(min(x, y), mul(mul(mul(h, h), h), mul(k, c(1.0f / 6.0f))));
    }
};

}

std::string GraphCompiler::socketType(const SocketRef socket) {
    return std::string(socketTypeName(socket->type));
}

std::string GraphCompiler::nodeName(NodeRef node) {
    return node->name;
}

void GraphCompiler::unsupported(NodeRef node, const std::string_view what) {
    diagnostics.unsupported(std::format("material '{}': the node '{}' {}", materialName, nodeName(node), what));
}

SocketRef GraphCompiler::input(NodeRef node, const std::string_view identifier) {
    if (const SocketRef socket = Materials::input(node, identifier))
        return socket;
    diagnostics.fail(std::format("material '{}': the node '{}' has no input {}", materialName, nodeName(node), identifier));
    return {};
}

//! The input that Blender passes to the output of a muted node, for a node that Blender did not mute.
SocketRef GraphCompiler::defaultInternalLink(const NodeRef node, const std::string_view identifier) {
    const NodeTypeData* type = findNodeTypeData(node->type);
    if (!type)
        return {};
    const auto output = static_cast<std::size_t>(std::ranges::find(type->outputs, identifier, &NodeSocketData::identifier) - type->outputs.begin());
    if (output >= type->muting.size())
        return {};
    const NodeMuting& muting = type->muting[output];
    std::vector<SocketRef> candidates;
    if (muting.unlinked >= 0)
        candidates.push_back(inputAt(node, static_cast<std::size_t>(muting.unlinked)));
    for (std::size_t index = 0; index < type->inputs.size(); ++index)
        if (muting.inputs >> index & 1 && static_cast<std::int32_t>(index) != muting.unlinked)
            candidates.push_back(inputAt(node, index));
    std::erase_if(candidates, [](const SocketRef socket) { return !socket || !socket->enabled; });
    const auto linked = std::ranges::find_if(candidates, [&](const SocketRef socket) { return links.isLinked(socket); });
    if (linked != candidates.end())
        return *linked;
    return candidates.empty() ? SocketRef{} : candidates.front();
}

SocketRef GraphCompiler::internalLink(const NodeRef node, const std::string_view identifier) {
    const std::span<const float> pairs = numbers(node, "internal_links");
    if (!findProperty(node, "internal_links"))
        return defaultInternalLink(node, identifier);
    for (std::size_t index = 0; index + 1 < pairs.size(); index += 2) {
        const SocketRef to = outputAt(node, static_cast<std::size_t>(pairs[index + 1]));
        if (to && to->identifier == identifier)
            return inputAt(node, static_cast<std::size_t>(pairs[index]));
    }
    return {};
}

std::uint32_t GraphCompiler::frameOf(const std::uint32_t parent, NodeRef group) {
    const auto [found, added] = frameKeys.try_emplace({parent, address(group)}, static_cast<std::uint32_t>(frames.size()));
    if (added)
        frames.push_back({parent, group});
    return found->second;
}

std::uint32_t GraphCompiler::textureSlot(const Scene::MaterialTexture& texture) {
    for (std::size_t slot = 0; slot < result.textures.size(); ++slot) {
        const Scene::MaterialTexture& existing = result.textures[slot];
        if (existing.texture == texture.texture && existing.wrap == texture.wrap && existing.filter == texture.filter)
            return static_cast<std::uint32_t>(slot);
    }
    result.textures.push_back(texture);
    return builder.addTexture();
}

GraphCompiler::Value GraphCompiler::zero(const std::string_view type) {
    return builder.resize(builder.constant(0.0f), widthOf(type));
}

GraphCompiler::Value GraphCompiler::parameter(const std::string& name, const NodeValues& value, const std::uint8_t width) {
    const Value result = builder.parameter(name, width);
    const std::array<float, 4>& v = value.values;
    builder.parameterValues().back() = value.count == 1 ? Core::float4{v[0], v[0], v[0], v[0]} : Core::float4{v[0], v[1], v[2], v[3]};
    return result;
}

GraphCompiler::Value GraphCompiler::convert(const Value value, const std::string_view from, const std::string_view to) {
    const std::uint8_t target = widthOf(to);
    if (builder.width(value) == target)
        return value;
    Ops o{builder};
    if (target == 1)
        return o.dot(o.rgb(value), o.c(from == "RGBA" ? luminanceWeights : averageWeights));
    return builder.resize(value, target);
}

GraphCompiler::Value GraphCompiler::value(const Context& context, NodeRef node, const std::string_view identifier) {
    const SocketRef socket = input(node, identifier);
    if (!socket)
        return builder.constant(0.0f);
    return socketValue(context, node, socket);
}

GraphCompiler::Value GraphCompiler::socketValue(const Context& context, NodeRef node, SocketRef socket) {
    const std::string type = socketType(socket);
    const NodeLinkEnd link = links.linkedNode(socket);
    if (!link.node)
        return defaultValue(context, node, socket, type);
    const std::string from = socketType(link.socket);
    if (from == "SHADER") {
        unsupported(link.node, "gives a shader to a value input");
        return zero(type);
    }
    return convert(output(context, link.node, link.socket), from, type);
}

GraphCompiler::Value GraphCompiler::defaultValue(const Context& context, NodeRef node, SocketRef socket, const std::string& type) {
    if (!isValueType(type)) {
        unsupported(node, std::format("has an unlinked input of the type {}", type));
        return builder.constant(0.0f);
    }
    const std::string identifier = socket->identifier;
    if (std::ranges::contains(numbers(node, "hide_value"), static_cast<float>(socket.index))) {
        if (identifier == "Normal" || identifier == "Coat Normal")
            return shadingNormal(context);
        if (identifier == "Tangent")
            return radialTangent(context, "Z");
        if (identifier == "Vector")
            return hit(context, GraphInput::Generated);
    }
    const NodeValues value = socketValues(socket);
    const std::string_view kind = nodeType(node);
    //! An input that decides whether a material is transparent or emissive, or whether a closure exists, is a constant at 0 and 1, so that the graph folds.
    const bool structural = (kind == "BSDF_PRINCIPLED"
                             && (identifier == "Alpha" || identifier == "Emission Strength" || identifier == "Coat Weight" || identifier == "Sheen Weight"
                                 || identifier == "Transmission Weight" || identifier == "Subsurface Weight"))
                            || (kind == "EMISSION" && identifier == "Strength") || (kind == "MIX_SHADER" && identifier == "Fac");
    if (structural && value.count >= 1 && (value.values[0] == 0.0f || value.values[0] == 1.0f))
        return builder.constant(value.values[0]);
    return parameter(std::format("{}.{}", nodeName(node), identifier), value, widthOf(type));
}

GraphCompiler::Value GraphCompiler::output(const Context& context, NodeRef node, SocketRef socket) {
    std::string identifier = socket->identifier;
    const Key key{context.frame, context.bump, address(node), identifier};
    if (const auto found = values.find(key); found != values.end())
        return found->second;
    const Value result = compileOutput(context, node, identifier, socketType(socket));
    values.emplace(key, result);
    return result;
}

//! World space inputs come in Blender's z-up space; the rows of the object-to-world transform swap their order instead, as they give world coordinates.
GraphCompiler::Value GraphCompiler::hit(const Context& context, const GraphInput which) {
    Ops o{builder};
    if (which == GraphInput::ObjectToWorld1)
        return o.neg(builder.input(GraphInput::ObjectToWorld2));
    if (which == GraphInput::ObjectToWorld2)
        return builder.input(GraphInput::ObjectToWorld1);
    Value base = builder.input(which);
    const bool x = context.bump.axis == 1;
    const auto offset = [&](const Value delta) { return o.mulAdd(delta, context.bump.scale, base); };
    if (context.bump.axis != 0) {
        if (which == GraphInput::Position)
            base = offset(builder.input(x ? GraphInput::DPdx : GraphInput::DPdy));
        else if (which == GraphInput::ObjectPosition)
            base = offset(builder.input(x ? GraphInput::ObjectDx : GraphInput::ObjectDy));
        else if (which == GraphInput::Generated)
            base = offset(o.mul(builder.input(x ? GraphInput::ObjectDx : GraphInput::ObjectDy), builder.input(GraphInput::GeneratedScale)));
        else if (which == GraphInput::Uv0 || which == GraphInput::Uv1 || which == GraphInput::Uv2 || which == GraphInput::Uv3) {
            const Value gradients = builder.input(GraphInput::UvGradients);
            base = offset(o.combine({o.x(gradients, x ? 0 : 2), o.x(gradients, x ? 1 : 3)}));
        }
    }
    switch (which) {
    case GraphInput::Position:
    case GraphInput::Normal:
    case GraphInput::GeometricNormal:
    case GraphInput::Tangent:
    case GraphInput::Incoming:
    case GraphInput::DPdx:
    case GraphInput::DPdy:
    case GraphInput::ObjectLocation:
    case GraphInput::WorldToObject0:
    case GraphInput::WorldToObject1:
    case GraphInput::WorldToObject2:
    case GraphInput::WorldToCamera0:
    case GraphInput::WorldToCamera1:
    case GraphInput::WorldToCamera2:
        return o.toBlender(base);
    default:
        return base;
    }
}

std::array<GraphCompiler::Value, 3> GraphCompiler::rows(const Context& context, const GraphInput first) {
    const auto at = [&](const std::uint8_t offset) { return hit(context, static_cast<GraphInput>(static_cast<std::uint8_t>(first) + offset)); };
    return {at(0), at(1), at(2)};
}

//! Cycles' radial tangent (svm_node_tangent): around the axis through the middle of the generated coordinates, turned like a normal, then normal to N.
GraphCompiler::Value GraphCompiler::radialTangent(const Context& context, const std::string_view axis) {
    Ops o{builder};
    const Value generated = o.sub(hit(context, GraphInput::Generated), o.c(0.5f));
    const Value zero = o.c(0.0f);
    Value local = o.combine({o.neg(o.x(generated, 1)), o.x(generated, 0), zero});
    if (axis == "X")
        local = o.combine({zero, o.neg(o.x(generated, 2)), o.x(generated, 1)});
    else if (axis == "Y")
        local = o.combine({o.neg(o.x(generated, 2)), zero, o.x(generated, 0)});
    const std::array<Value, 3> toObject = rows(context, GraphInput::WorldToObject0);
    const Value world = o.mulAdd(o.rgb(toObject[0]), o.x(local, 0), o.mulAdd(o.rgb(toObject[1]), o.x(local, 1), o.mul(o.rgb(toObject[2]), o.x(local, 2))));
    const Value normal = hit(context, GraphInput::Normal);
    return o.cross(normal, o.normalize(o.cross(world, normal)));
}

//! The Tangent input of an anisotropic BSDF: its link, else the radial tangent about Z that Cycles links to it; without anisotropy it is unused.
GraphCompiler::Value GraphCompiler::anisotropicTangent(const Context& context, NodeRef node, const Value anisotropic) {
    if (const SocketRef tangent = input(node, "Tangent"); tangent && links.linkedNode(tangent).node)
        return Ops{builder}.normalize(socketValue(context, node, tangent));
    if (builder.constantOf(anisotropic) && builder.constantOf(anisotropic)[0] == 0.0f)
        return builder.resize(hit(context, GraphInput::Tangent), 3);
    return radialTangent(context, "Z");
}

static AttributeScope toAttributeScope(const std::string_view scope) {
    return scope == "INSTANCER" ? AttributeScope::Instancer : scope == "VIEW_LAYER" ? AttributeScope::ViewLayer : AttributeScope::Object;
}

GraphCompiler::Value GraphCompiler::uv(const Context& context, const std::string_view map) {
    if (map.empty() || !attributeTables)
        return builder.resize(hit(context, GraphInput::Uv0), 3);
    return builder.resize(stream(context, std::string(map), UvPart::Low), 3);
}

GraphCompiler::Value GraphCompiler::stream(const Context& context, const std::string& name, const UvPart part) {
    const std::optional<std::size_t> set = attributeTables ? attributeTables->uvStreams.assign(result.streams, {name, part}) : std::nullopt;
    if (!set) {
        diagnostics.unsupported(std::format("material '{}': the attribute '{}' renders as 0, because a material reads at most three pairs of values from attributes",
                                            materialName, name));
        return builder.resize(zero("VALUE"), 2);
    }
    constexpr std::array sets{GraphInput::Uv0, GraphInput::Uv1, GraphInput::Uv2, GraphInput::Uv3};
    return hit(context, sets[*set]);
}

GraphCompiler::Value GraphCompiler::compileOutput(const Context& context, NodeRef node, const std::string& identifier, const std::string& type) {
    Ops o{builder};
    const std::string_view kind = nodeType(node);
    if (diagnostics.failed())
        return zero(type);
    if (node->muted) {
        if (const SocketRef from = internalLink(node, identifier))
            return convert(socketValue(context, node, from), socketType(from), type);
        return zero(type);
    }
    if (kind == "GROUP_INPUT") {
        if (context.frame == 0) {
            unsupported(node, "is a group input outside of a group");
            return zero(type);
        }
        const Frame frame = frames[context.frame];
        const SocketRef socket = input(frame.group, identifier);
        if (!socket)
            return zero(type);
        return convert(socketValue({frame.parent, context.bump}, frame.group, socket), socketType(socket), type);
    }
    if (kind == "GROUP") {
        const NodeTree* tree = groupTree(node);
        const NodeRef groupOutput = tree ? findGroupOutput(*tree) : NodeRef{};
        if (!groupOutput) {
            unsupported(node, "is a group without an output");
            return zero(type);
        }
        const SocketRef socket = input(groupOutput, identifier);
        if (!socket)
            return zero(type);
        return convert(socketValue({frameOf(context.frame, node), context.bump}, groupOutput, socket), socketType(socket), type);
    }
    const auto in = [&](const std::string_view name) { return value(context, node, name); };
    if (kind == "VALUE" || kind == "RGB") {
        const SocketRef socket = outputAt(node, 0);
        return parameter(std::format("{}.{}", nodeName(node), identifier), socketValues(socket), widthOf(type));
    }
    if (kind == "MATH")
        return math(context, node);
    if (kind == "VECT_MATH")
        return vectorMath(context, node, identifier);
    if (kind == "MIX")
        return mix(context, node, identifier);
    if (kind == "MIX_RGB")
        return mixRgb(context, node);
    if (kind == "MAP_RANGE")
        return mapRange(context, node, identifier);
    if (kind == "CLAMP") {
        if (text(node, "clamp_type") == "RANGE")
            return builder.op(GraphOp::Clamp, in("Value"), in("Min"), in("Max"));
        return o.min(o.max(in("Value"), in("Min")), in("Max"));
    }
    if (kind == "VALTORGB")
        return colorRamp(context, node, identifier);
    if (kind == "CURVE_RGB" || kind == "CURVE_VEC" || kind == "CURVE_FLOAT")
        return curve(context, node);
    if (kind == "SEPARATE_COLOR")
        return separateColor(context, node, identifier);
    if (kind == "COMBINE_COLOR")
        return combineColor(context, node);
    if (kind == "SEPXYZ")
        return o.x(in("Vector"), identifier == "X" ? 0 : identifier == "Y" ? 1 : 2);
    if (kind == "COMBXYZ")
        return o.combine({in("X"), in("Y"), in("Z")});
    if (kind == "TEX_COORD")
        return texCoord(context, node, identifier);
    if (kind == "NEW_GEOMETRY")
        return geometry(context, node, identifier);
    if (kind == "WIREFRAME") {
        if (flag(node, "use_pixel_size"))
            unsupported(node, "has a pixel size, which renders in Blender units");
        return o.call(GraphFunction::Wireframe, {o.toNoorRay(hit(context, GraphInput::Position)), in("Size")});
    }
    if (kind == "OBJECT_INFO")
        return objectInfo(context, node, identifier);
    if (kind == "CAMERA")
        return camera(context, node, identifier);
    if (kind == "LIGHT_FALLOFF")
        return lightFalloff(context, node, identifier);
    //! On a mesh, Cycles gives 0 for every output of the nodes of curves, points, particles and volume grids.
    if (kind == "HAIR_INFO" || kind == "POINT_INFO" || kind == "PARTICLE_INFO" || (kind == "VOLUME_INFO" && !volume))
        return zero(type);
    if (kind == "ATTRIBUTE" || kind == "VERTEX_COLOR") {
        const std::string name = text(node, kind == "ATTRIBUTE" ? "attribute_name" : "layer_name");
        const std::string scope = kind == "ATTRIBUTE" ? text(node, "attribute_type") : std::string("GEOMETRY");
        if (scope != "GEOMETRY" && (name.empty() || !attributeTables))
            return zero(type);
        if (scope != "GEOMETRY") {
            const std::size_t slot = attributeTables->instances.assign({name, toAttributeScope(scope)});
            const Value attribute = o.call(GraphFunction::InstanceAttribute, {o.c(static_cast<float>(slot))});
            if (identifier == "Alpha")
                return o.x(attribute, 3);
            return identifier == "Fac" ? o.dot(o.rgb(attribute), o.c(averageWeights)) : o.rgb(attribute);
        }
        if (!name.empty()) {
            if (identifier == "Fac" || identifier == "Alpha")
                return o.x(stream(context, name, UvPart::Mean), identifier == "Fac" ? 0 : 1);
            const Value low = stream(context, name, UvPart::Low);
            return o.combine({o.x(low, 0), o.x(low, 1), o.x(stream(context, name, UvPart::High), 0)});
        }
        if (kind == "ATTRIBUTE")
            return zero(type);
        result.vertexColors = true;
        const Value color = hit(context, GraphInput::Color);
        if (identifier == "Alpha")
            return o.x(color, 3);
        if (identifier == "Fac")
            return o.dot(o.rgb(color), o.c(averageWeights));
        return o.rgb(color);
    }
    if (kind == "UVMAP")
        return uv(context, text(node, "uv_map"));
    if (kind == "NORMAL_MAP")
        return normalMap(context, node);
    if (kind == "BUMP")
        return bump(context, node);
    if (kind == "MAPPING")
        return mapping(context, node);
    if (kind == "VECTOR_ROTATE")
        return vectorRotate(context, node);
    if (kind == "VECT_TRANSFORM")
        return vectorTransform(context, node);
    if (kind == "DISPLACEMENT")
        return displacement(context, node);
    if (kind == "VECTOR_DISPLACEMENT")
        return vectorDisplacement(context, node);
    if (kind == "FRESNEL")
        return o.call(GraphFunction::Fresnel,
                      {in("IOR"), normalInput(context, node, "Normal"), o.rgb(hit(context, GraphInput::Incoming)), hit(context, GraphInput::Backfacing)});
    if (kind == "LAYER_WEIGHT") {
        const Value weights = o.call(GraphFunction::LayerWeight, {in("Blend"), normalInput(context, node, "Normal"), o.rgb(hit(context, GraphInput::Incoming)),
                                                                  hit(context, GraphInput::Backfacing)});
        return o.x(weights, identifier == "Fresnel" ? 0 : 1);
    }
    if (kind == "HUE_SAT")
        return o.call(GraphFunction::HueSaturationValue, {in("Color"), o.combine({in("Hue"), in("Saturation"), in("Value"), in("Fac")})});
    if (kind == "BLACKBODY")
        return o.call(GraphFunction::Blackbody, {in("Temperature")});
    if (kind == "WAVELENGTH")
        return o.call(GraphFunction::Wavelength, {in("Wavelength")});
    if (kind == "INVERT") {
        const Value color = in("Color");
        return o.mix(color, o.sub(o.c(1.0f), color), in("Fac"));
    }
    if (kind == "GAMMA") {
        const Value color = in("Color");
        return o.select(o.greater(color, o.c(0.0f)), builder.op(GraphOp::Pow, color, in("Gamma")), color);
    }
    if (kind == "BRIGHTCONTRAST") {
        const Value contrast = in("Contrast");
        const Value offset = o.sub(in("Bright"), o.mul(contrast, o.c(0.5f)));
        return o.max(o.mulAdd(in("Color"), o.add(contrast, o.c(1.0f)), offset), o.c(0.0f));
    }
    if (kind == "RGBTOBW")
        return o.dot(in("Color"), o.c(luminanceWeights));
    if (kind == "NORMAL") {
        const NodeValues stored = socketValues(outputAt(node, 0));
        const Core::float3 direction = Core::normalize(Core::float3{stored.values[0], stored.values[1], stored.values[2]});
        if (identifier == "Normal")
            return o.c(direction);
        return o.dot(o.c(direction), o.normalize(in("Normal")));
    }
    if (kind == "TANGENT") {
        if (text(node, "direction_type") != "UV_MAP")
            return radialTangent(context, text(node, "axis"));
        result.tangents = true;
        return o.rgb(hit(context, GraphInput::Tangent));
    }
    if (kind == "LIGHT_PATH")
        return lightPath(context, node, identifier);
    if (kind == "AMBIENT_OCCLUSION")
        return ambientOcclusion(context, node, identifier);
    if (kind == "BEVEL")
        return bevel(context, node);
    if (kind == "MATERIAL_RAYCAST")
        return raycast(context, node, identifier);
    if (kind == "TEX_IMAGE")
        return image(context, node, identifier);
    if (kind == "TEX_ENVIRONMENT")
        return environment(context, node, identifier);
    if (kind == "TEX_NOISE")
        return noise(context, node, identifier);
    if (kind == "TEX_VORONOI")
        return voronoi(context, node, identifier);
    if (kind == "TEX_WAVE")
        return wave(context, node, identifier);
    if (kind == "TEX_MAGIC")
        return magic(context, node, identifier);
    if (kind == "TEX_CHECKER")
        return checker(context, node, identifier);
    if (kind == "TEX_BRICK")
        return brick(context, node, identifier);
    if (kind == "TEX_GRADIENT")
        return gradient(context, node, identifier);
    if (kind == "TEX_WHITE_NOISE")
        return whiteNoise(context, node, identifier);
    if (kind == "TEX_GABOR")
        return gabor(context, node, identifier);
    if (kind == "TEX_IES")
        return ies(context, node);
    if (kind == "TEX_SKY")
        return sky(context, node);
    if (kind == "ShaderNodeRadialTiling")
        return radialTiling(context, node, identifier);
    unsupported(node, std::format("({}) has no graph translation", kind));
    return zero(type);
}

GraphCompiler::Value GraphCompiler::texCoord(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    if (identifier == "Generated")
        return hit(context, GraphInput::Generated);
    if (identifier == "UV")
        return uv(context, "");
    if (identifier == "Normal") {
        const std::array<Value, 3> toWorld = rows(context, GraphInput::ObjectToWorld0);
        const Value normal = hit(context, GraphInput::Normal);
        return o.normalize(o.mulAdd(o.rgb(toWorld[0]), o.x(normal, 0), o.mulAdd(o.rgb(toWorld[1]), o.x(normal, 1), o.mul(o.rgb(toWorld[2]), o.x(normal, 2)))));
    }
    if (identifier == "Camera")
        return o.transform(rows(context, GraphInput::WorldToCamera0), hit(context, GraphInput::Position), true);
    if (identifier == "Window") {
        const Value view = o.transform(rows(context, GraphInput::WorldToCamera0), hit(context, GraphInput::Position), true);
        const Value flat = o.combine({o.x(view, 0), o.x(view, 1)});
        const Value projected = o.select(hit(context, GraphInput::CameraPerspective), o.div(flat, o.x(view, 2)), flat);
        const Value window = hit(context, GraphInput::CameraWindow);
        const Value ndc = o.mulAdd(projected, o.combine({o.x(window, 0), o.x(window, 1)}), o.combine({o.x(window, 2), o.x(window, 3)}));
        return o.combine({ndc, o.c(0.0f)});
    }
    if (identifier == "Object") {
        const std::span<const float> inverse = numbers(node, "object_inverse");
        if (inverse.size() < 12)
            return hit(context, GraphInput::ObjectPosition);
        const Value position = hit(context, GraphInput::Position);
        std::array<Value, 3> rows{};
        for (std::size_t row = 0; row < 3; ++row) {
            const std::span<const float> values = inverse.subspan(row * 4, 4);
            rows[row] = o.add(o.dot(o.c(Core::float3{values[0], values[1], values[2]}), position), o.c(values[3]));
        }
        return o.combine({rows[0], rows[1], rows[2]});
    }
    if (identifier == "Reflection")
        return builder.op(GraphOp::Reflect, o.neg(o.rgb(hit(context, GraphInput::Incoming))), hit(context, GraphInput::Normal));
    unsupported(node, std::format("gives the {} coordinates, which a material graph does not know", identifier));
    return zero("VECTOR");
}

//! Cycles' svm_node_camera: the hit in camera space, z forward.
GraphCompiler::Value GraphCompiler::camera(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    const Value view = o.transform(rows(context, GraphInput::WorldToCamera0), hit(context, GraphInput::Position), true);
    if (identifier == "View Vector")
        return o.normalize(view);
    if (identifier == "View Z Depth")
        return o.x(view, 2);
    if (identifier == "View Distance")
        return builder.op(GraphOp::Length, view);
    unsupported(node, std::format("gives {}, which a material graph does not know", identifier));
    return zero("VALUE");
}

GraphCompiler::Value GraphCompiler::lightPath(const Context& context, NodeRef node, const std::string& identifier) {
    static constexpr std::array<std::pair<std::string_view, GraphInput>, 13> outputs{{
        {"Is Camera Ray", GraphInput::IsCameraRay},
        {"Is Shadow Ray", GraphInput::IsShadowRay},
        {"Is Diffuse Ray", GraphInput::IsDiffuseRay},
        {"Is Glossy Ray", GraphInput::IsGlossyRay},
        {"Is Singular Ray", GraphInput::IsSingularRay},
        {"Is Reflection Ray", GraphInput::IsReflectionRay},
        {"Is Transmission Ray", GraphInput::IsTransmissionRay},
        {"Is Volume Scatter Ray", GraphInput::IsVolumeScatterRay},
        {"Ray Length", GraphInput::Distance},
        {"Ray Depth", GraphInput::RayDepth},
        {"Diffuse Depth", GraphInput::DiffuseDepth},
        {"Glossy Depth", GraphInput::GlossyDepth},
        {"Transmission Depth", GraphInput::TransmissionDepth},
    }};
    for (const auto& [name, input] : outputs)
        if (name == identifier)
            return hit(context, input);
    if (identifier == "Transparent Depth")
        return hit(context, GraphInput::TransparentDepth);
    //! NoorRay has no portals, so a ray never passed one.
    if (identifier == "Portal Depth")
        return builder.constant(0.0f);
    unsupported(node, std::format("gives {}, which a material graph does not know", identifier));
    return zero("VALUE");
}

bool GraphCompiler::linked(NodeRef node, const std::string_view identifier) {
    const SocketRef socket = input(node, identifier);
    return socket && links.linkedNode(socket).node;
}

//! Cycles' svm_node_ao: the rays leave the hit along a cosine lobe of the normal, and the colour output is the colour times the AO.
GraphCompiler::Value GraphCompiler::ambientOcclusion(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    const SocketRef distanceSocket = input(node, "Distance");
    const bool global = !linked(node, "Distance") && socketValues(distanceSocket).values[0] == 0.0f;
    const Value distance = global ? o.c(number(node->properties, "world_distance", 10.0f)) : value(context, node, "Distance");
    const Value settings = o.combine({distance, o.c(static_cast<float>(integer(node, "samples"))), o.c(flag(node, "inside") ? 1.0f : 0.0f),
                                      o.c(flag(node, "only_local") ? 1.0f : 0.0f)});
    const Value ao =
        o.call(GraphFunction::AmbientOcclusion, {o.toNoorRay(hit(context, GraphInput::Position)), o.toNoorRay(normalInput(context, node, "Normal")), settings});
    if (identifier == "AO")
        return ao;
    return o.mul(o.rgb(value(context, node, "Color")), ao);
}

//! Cycles' svm_node_bevel: it skips rays after a diffuse bounce, as Cycles does below a ray pdf of 8, and a linked normal keeps its offset from the hit normal.
GraphCompiler::Value GraphCompiler::bevel(const Context& context, NodeRef node) {
    Ops o{builder};
    const Value normal = shadingNormal(context);
    const Value skip = o.greater(hit(context, GraphInput::DiffuseDepth), o.c(0.0f));
    const Value settings =
        o.combine({value(context, node, "Radius"), o.c(static_cast<float>(integer(node, "samples"))), hit(context, GraphInput::Backfacing), skip});
    const Value bevelled = o.toBlender(o.call(GraphFunction::Bevel, {o.toNoorRay(hit(context, GraphInput::Position)),
                                                                     o.toNoorRay(hit(context, GraphInput::GeometricNormal)), o.toNoorRay(normal), settings}));
    if (!linked(node, "Normal"))
        return bevelled;
    return o.normalize(o.add(value(context, node, "Normal"), o.sub(bevelled, normal)));
}

//! Cycles' svm_node_raycast: a ray from the hit itself skips the hit triangle, and a miss gives the length as its distance.
GraphCompiler::Value GraphCompiler::raycast(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    if (integer(node, "sample_attribute_items") != 0)
        unsupported(node, "samples attributes at the hit");
    const bool fromHit = !linked(node, "Position");
    const Value position = fromHit ? hit(context, GraphInput::Position) : value(context, node, "Position");
    const Value direction = linked(node, "Direction") ? value(context, node, "Direction") : shadingNormal(context);
    const Value settings =
        o.combine({value(context, node, "Length"), o.c(flag(node, "only_local") ? 1.0f : 0.0f), o.c(fromHit ? 1.0f : 0.0f), o.c(0.0f)});
    if (identifier == "Hit Normal")
        return o.toBlender(o.call(GraphFunction::RaycastNormal, {o.toNoorRay(position), o.toNoorRay(direction), settings}));
    const Value traced = o.call(GraphFunction::Raycast, {o.toNoorRay(position), o.toNoorRay(direction), settings});
    if (identifier == "Is Hit")
        return o.x(traced, 0);
    if (identifier == "Self Hit")
        return o.x(traced, 1);
    if (identifier == "Hit Distance")
        return o.x(traced, 2);
    return o.mul(o.x(traced, 0), o.mulAdd(direction, o.x(traced, 2), position));
}

GraphCompiler::Value GraphCompiler::objectInfo(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    if (identifier == "Location")
        return hit(context, GraphInput::ObjectLocation);
    if (identifier == "Random")
        return hit(context, GraphInput::ObjectRandom);
    if (identifier == "Color")
        return o.rgb(hit(context, GraphInput::ObjectColor));
    if (identifier == "Alpha")
        return o.x(hit(context, GraphInput::ObjectColor), 3);
    if (identifier == "SocketRef Index")
        return hit(context, GraphInput::ObjectIndex);
    if (identifier == "Material Index")
        return builder.constant(materialIndex);
    unsupported(node, std::format("gives the object {}, which NoorRay does not keep per instance", identifier));
    return zero("VALUE");
}

//! Cycles' svm_node_vector_transform: a point moves with the translation, a vector does not, and a normal turns by the transpose of the inverse.
GraphCompiler::Value GraphCompiler::vectorTransform(const Context& context, NodeRef node) {
    Ops o{builder};
    const std::string from = text(node, "convert_from");
    const std::string to = text(node, "convert_to");
    const std::string type = text(node, "vector_type");
    Value vector = value(context, node, "Vector");
    if (from == to)
        return vector;
    if (type == "NORMAL") {
        if (from == "OBJECT")
            vector = o.normalize(o.transposed(rows(context, GraphInput::WorldToObject0), vector));
        else if (from == "CAMERA")
            vector = o.normalize(o.transposed(rows(context, GraphInput::WorldToCamera0), vector));
        if (to == "OBJECT")
            return o.normalize(o.transposed(rows(context, GraphInput::ObjectToWorld0), vector));
        if (to == "CAMERA")
            return o.normalize(o.inverseTransposed(rows(context, GraphInput::WorldToCamera0), vector));
        return vector;
    }
    const bool point = type == "POINT";
    if (from == "OBJECT")
        vector = o.transform(rows(context, GraphInput::ObjectToWorld0), vector, point);
    else if (from == "CAMERA")
        vector = o.inverseTransform(rows(context, GraphInput::WorldToCamera0), vector, point);
    if (to == "OBJECT")
        return o.transform(rows(context, GraphInput::WorldToObject0), vector, point);
    if (to == "CAMERA")
        return o.transform(rows(context, GraphInput::WorldToCamera0), vector, point);
    return vector;
}

//! Cycles' svm_node_light_falloff with the ray length of the hit.
GraphCompiler::Value GraphCompiler::lightFalloff(const Context& context, NodeRef node, const std::string& identifier) {
    if (light) {
        if (linked(node, "Smooth"))
            unsupported(node, "has a linked Smooth in a light shader");
        result.falloffPower = identifier == "Constant" ? 2u : (identifier == "Linear" ? 1u : 0u);
        result.falloffSmooth = std::max(socketNumber(input(node, "Smooth")), 0.0f);
        return value(context, node, "Strength");
    }
    Ops o{builder};
    const Value length = hit(context, GraphInput::Distance);
    const Value squared = o.mul(length, length);
    Value strength = value(context, node, "Strength");
    if (identifier == "Linear")
        strength = o.mul(strength, length);
    else if (identifier == "Constant")
        strength = o.mul(strength, squared);
    const Value smooth = value(context, node, "Smooth");
    return o.select(o.greater(smooth, o.c(0.0f)), o.mul(strength, o.div(squared, o.add(smooth, squared))), strength);
}

GraphCompiler::Value GraphCompiler::geometry(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    if (identifier == "Position")
        return hit(context, GraphInput::Position);
    if (identifier == "Normal")
        return hit(context, GraphInput::Normal);
    if (identifier == "True Normal")
        return hit(context, GraphInput::GeometricNormal);
    if (identifier == "Incoming")
        return o.rgb(hit(context, GraphInput::Incoming));
    if (identifier == "Backfacing")
        return hit(context, GraphInput::Backfacing);
    if (identifier == "Tangent") {
        result.tangents = true;
        return o.rgb(hit(context, GraphInput::Tangent));
    }
    if (identifier == "Pointiness" || identifier == "Random Per Island")
        return o.x(stream(context, identifier == "Pointiness" ? pointinessStream : islandStream, UvPart::Mean), 0);
    if (identifier == "Parametric") {
        const Value weights = o.call(GraphFunction::Parametric, {o.toNoorRay(hit(context, GraphInput::Position))});
        return o.combine({o.x(weights, 0), o.x(weights, 1), o.c(0.0f)});
    }
    unsupported(node, std::format("gives {}, which a material graph does not know", identifier));
    return zero(identifier == "Parametric" ? "VECTOR" : "VALUE");
}

GraphCompiler::Value GraphCompiler::math(const Context& context, NodeRef node) {
    Ops o{builder};
    const std::string operation = text(node, "operation");
    const Value a = value(context, node, "Value");
    const auto b = [&] { return value(context, node, "Value_001"); };
    const auto c = [&] { return value(context, node, "Value_002"); };
    const auto unary = [&](const GraphOp op) { return builder.op(op, a); };
    const auto binary = [&](const GraphOp op) { return builder.op(op, a, b()); };
    Value r = a;
    if (operation == "ADD")
        r = binary(GraphOp::Add);
    else if (operation == "SUBTRACT")
        r = binary(GraphOp::Sub);
    else if (operation == "MULTIPLY")
        r = binary(GraphOp::Mul);
    else if (operation == "DIVIDE")
        r = binary(GraphOp::Div);
    else if (operation == "MULTIPLY_ADD")
        r = o.mulAdd(a, b(), c());
    else if (operation == "POWER")
        r = binary(GraphOp::Pow);
    else if (operation == "LOGARITHM")
        r = o.div(unary(GraphOp::Log), builder.op(GraphOp::Log, b()));
    else if (operation == "SQRT")
        r = unary(GraphOp::Sqrt);
    else if (operation == "INVERSE_SQRT")
        r = unary(GraphOp::InverseSqrt);
    else if (operation == "ABSOLUTE")
        r = unary(GraphOp::Abs);
    else if (operation == "EXPONENT")
        r = unary(GraphOp::Exp);
    else if (operation == "MINIMUM")
        r = binary(GraphOp::Min);
    else if (operation == "MAXIMUM")
        r = binary(GraphOp::Max);
    else if (operation == "LESS_THAN")
        r = binary(GraphOp::Less);
    else if (operation == "GREATER_THAN")
        r = binary(GraphOp::Greater);
    else if (operation == "SIGN")
        r = unary(GraphOp::Sign);
    else if (operation == "COMPARE")
        r = o.sub(o.c(1.0f), o.greater(o.abs(o.sub(a, b())), o.max(c(), o.c(1e-5f))));
    else if (operation == "SMOOTH_MIN")
        r = o.smoothMin(a, b(), c());
    else if (operation == "SMOOTH_MAX")
        r = o.neg(o.smoothMin(o.neg(a), o.neg(b()), c()));
    else if (operation == "ROUND")
        r = o.floor(o.add(a, o.c(0.5f)));
    else if (operation == "FLOOR")
        r = unary(GraphOp::Floor);
    else if (operation == "CEIL")
        r = unary(GraphOp::Ceil);
    else if (operation == "TRUNC")
        r = unary(GraphOp::Trunc);
    else if (operation == "FRACT")
        r = unary(GraphOp::Fract);
    else if (operation == "MODULO")
        r = binary(GraphOp::Modulo);
    else if (operation == "FLOORED_MODULO")
        r = binary(GraphOp::FlooredModulo);
    else if (operation == "WRAP")
        r = o.wrap(a, b(), c());
    else if (operation == "SNAP") {
        const Value step = b();
        r = o.mul(o.floor(o.div(a, step)), step);
    } else if (operation == "PINGPONG") {
        const Value scale = b();
        const Value twice = o.mul(scale, o.c(2.0f));
        const Value folded = o.abs(o.sub(o.mul(builder.op(GraphOp::Fract, o.div(o.sub(a, scale), twice)), twice), scale));
        r = o.select(o.greater(o.abs(scale), o.c(0.0f)), folded, o.c(0.0f));
    } else if (operation == "SINE")
        r = unary(GraphOp::Sin);
    else if (operation == "COSINE")
        r = unary(GraphOp::Cos);
    else if (operation == "TANGENT")
        r = unary(GraphOp::Tan);
    else if (operation == "ARCSINE")
        r = unary(GraphOp::Asin);
    else if (operation == "ARCCOSINE")
        r = unary(GraphOp::Acos);
    else if (operation == "ARCTANGENT")
        r = unary(GraphOp::Atan);
    else if (operation == "ARCTAN2")
        r = binary(GraphOp::Atan2);
    else if (operation == "SINH")
        r = unary(GraphOp::Sinh);
    else if (operation == "COSH")
        r = unary(GraphOp::Cosh);
    else if (operation == "TANH")
        r = unary(GraphOp::Tanh);
    else if (operation == "RADIANS")
        r = o.mul(a, o.c(std::numbers::pi_v<float> / 180.0f));
    else if (operation == "DEGREES")
        r = o.mul(a, o.c(180.0f / std::numbers::pi_v<float>));
    else
        unsupported(node, std::format("uses the math operation {}", operation));
    return flag(node, "use_clamp") ? o.saturate(r) : r;
}

GraphCompiler::Value GraphCompiler::vectorMath(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    const std::string operation = text(node, "operation");
    const Value a = value(context, node, "Vector");
    const auto b = [&] { return value(context, node, "Vector_001"); };
    const auto c = [&] { return value(context, node, "Vector_002"); };
    const auto unary = [&](const GraphOp op) { return builder.op(op, a); };
    const auto binary = [&](const GraphOp op) { return builder.op(op, a, b()); };
    const bool scalar = operation == "DOT_PRODUCT" || operation == "DISTANCE" || operation == "LENGTH";
    if (scalar != (identifier == "Value"))
        return zero(identifier == "Value" ? "VALUE" : "VECTOR");
    if (operation == "ADD")
        return binary(GraphOp::Add);
    if (operation == "SUBTRACT")
        return binary(GraphOp::Sub);
    if (operation == "MULTIPLY")
        return binary(GraphOp::Mul);
    if (operation == "DIVIDE")
        return binary(GraphOp::Div);
    if (operation == "MULTIPLY_ADD")
        return o.mulAdd(a, b(), c());
    if (operation == "CROSS_PRODUCT")
        return binary(GraphOp::Cross);
    if (operation == "PROJECT") {
        const Value onto = b();
        return o.mul(o.div(o.dot(a, onto), o.dot(onto, onto)), onto);
    }
    if (operation == "REFLECT")
        return builder.op(GraphOp::Reflect, a, o.normalize(b()));
    if (operation == "REFRACT")
        return builder.op(GraphOp::Refract, a, o.normalize(b()), value(context, node, "Scale"));
    if (operation == "FACEFORWARD")
        return builder.op(GraphOp::FaceForward, a, b(), c());
    if (operation == "DOT_PRODUCT")
        return binary(GraphOp::Dot);
    if (operation == "DISTANCE")
        return binary(GraphOp::Distance);
    if (operation == "LENGTH")
        return unary(GraphOp::Length);
    if (operation == "SCALE")
        return o.mul(a, value(context, node, "Scale"));
    if (operation == "NORMALIZE")
        return unary(GraphOp::Normalize);
    if (operation == "ABSOLUTE")
        return unary(GraphOp::Abs);
    if (operation == "POWER")
        return binary(GraphOp::Pow);
    if (operation == "SIGN")
        return unary(GraphOp::Sign);
    if (operation == "MINIMUM")
        return binary(GraphOp::Min);
    if (operation == "MAXIMUM")
        return binary(GraphOp::Max);
    if (operation == "ROUND")
        return unary(GraphOp::Round);
    if (operation == "FLOOR")
        return unary(GraphOp::Floor);
    if (operation == "CEIL")
        return unary(GraphOp::Ceil);
    if (operation == "FRACTION")
        return unary(GraphOp::Fract);
    if (operation == "MODULO")
        return binary(GraphOp::Modulo);
    if (operation == "WRAP")
        return o.wrap(a, b(), c());
    if (operation == "SNAP") {
        const Value step = b();
        return o.mul(o.floor(o.div(a, step)), step);
    }
    if (operation == "SINE")
        return unary(GraphOp::Sin);
    if (operation == "COSINE")
        return unary(GraphOp::Cos);
    if (operation == "TANGENT")
        return unary(GraphOp::Tan);
    unsupported(node, std::format("uses the vector math operation {}", operation));
    return a;
}

GraphCompiler::Value GraphCompiler::blend(const std::string& mode, const Value t, const Value a, const Value b) {
    Ops o{builder};
    const Value one = o.c(1.0f);
    const Value zeroValue = o.c(0.0f);
    const Value tm = o.sub(one, t);
    const auto hsv = [&](const Value color) { return o.call(GraphFunction::RgbToHsv, {color}); };
    const auto rgb = [&](const Value color) { return o.call(GraphFunction::HsvToRgb, {color}); };
    if (mode == "MIX")
        return o.mix(a, b, t);
    if (mode == "ADD")
        return o.mulAdd(b, t, a);
    if (mode == "MULTIPLY")
        return o.mix(a, o.mul(a, b), t);
    if (mode == "SUBTRACT")
        return o.sub(a, o.mul(b, t));
    if (mode == "SCREEN")
        return o.sub(one, o.mul(o.add(tm, o.mul(t, o.sub(one, b))), o.sub(one, a)));
    if (mode == "OVERLAY") {
        const Value low = o.mul(a, o.add(tm, o.mul(o.mul(t, o.c(2.0f)), b)));
        const Value high = o.sub(one, o.mul(o.add(tm, o.mul(o.mul(t, o.c(2.0f)), o.sub(one, b))), o.sub(one, a)));
        return o.select(o.less(a, o.c(0.5f)), low, high);
    }
    if (mode == "DIVIDE")
        return o.select(o.greater(o.abs(b), zeroValue), o.add(o.mul(tm, a), o.mul(t, o.div(a, b))), a);
    if (mode == "DIFFERENCE")
        return o.mix(a, o.abs(o.sub(a, b)), t);
    if (mode == "EXCLUSION")
        return o.max(o.mix(a, o.sub(o.add(a, b), o.mul(o.mul(a, b), o.c(2.0f))), t), zeroValue);
    if (mode == "DARKEN")
        return o.mix(a, o.min(a, b), t);
    if (mode == "LIGHTEN")
        return o.mix(a, o.max(a, b), t);
    if (mode == "DODGE") {
        const Value rest = o.sub(one, o.mul(t, b));
        const Value dodged = o.select(o.greater(rest, zeroValue), o.min(o.div(a, rest), one), one);
        return o.select(o.greater(o.abs(a), zeroValue), dodged, a);
    }
    if (mode == "BURN") {
        const Value rest = o.add(tm, o.mul(t, b));
        return o.select(o.greater(rest, zeroValue), o.saturate(o.sub(one, o.div(o.sub(one, a), rest))), zeroValue);
    }
    if (mode == "SOFT_LIGHT") {
        const Value screen = o.sub(one, o.mul(o.sub(one, b), o.sub(one, a)));
        return o.add(o.mul(tm, a), o.mul(t, o.add(o.mul(o.mul(o.sub(one, a), b), a), o.mul(a, screen))));
    }
    if (mode == "LINEAR_LIGHT")
        return o.add(a, o.mul(t, o.sub(o.mul(b, o.c(2.0f)), one)));
    const Value hsvA = hsv(a);
    const Value hsvB = hsv(b);
    const Value hasSaturation = o.greater(o.x(hsvB, 1), zeroValue);
    if (mode == "HUE")
        return o.select(hasSaturation, o.mix(a, rgb(o.combine({o.x(hsvB, 0), o.x(hsvA, 1), o.x(hsvA, 2)})), t), a);
    if (mode == "SATURATION") {
        const Value saturation = o.mix(o.x(hsvA, 1), o.x(hsvB, 1), t);
        return o.select(o.greater(o.x(hsvA, 1), zeroValue), rgb(o.combine({o.x(hsvA, 0), saturation, o.x(hsvA, 2)})), a);
    }
    if (mode == "VALUE")
        return rgb(o.combine({o.x(hsvA, 0), o.x(hsvA, 1), o.mix(o.x(hsvA, 2), o.x(hsvB, 2), t)}));
    if (mode == "COLOR")
        return o.select(hasSaturation, o.mix(a, rgb(o.combine({o.x(hsvB, 0), o.x(hsvB, 1), o.x(hsvA, 2)})), t), a);
    diagnostics.unsupported(std::format("material '{}': the blend mode {} has no graph translation", materialName, mode));
    return o.mix(a, b, t);
}

GraphCompiler::Value GraphCompiler::mix(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    const std::string type = text(node, "data_type");
    const auto factor = [&](const std::string_view name) {
        const Value value = this->value(context, node, name);
        return flag(node, "clamp_factor") ? o.saturate(value) : value;
    };
    if (type == "FLOAT")
        return o.mix(value(context, node, "A_Float"), value(context, node, "B_Float"), factor("Factor_Float"));
    if (type == "VECTOR") {
        const Value t = factor(text(node, "factor_mode") == "NON_UNIFORM" ? "Factor_Vector" : "Factor_Float");
        return o.mix(value(context, node, "A_Vector"), value(context, node, "B_Vector"), t);
    }
    if (type == "RGBA") {
        const Value blended = blend(text(node, "blend_type"), factor("Factor_Float"), value(context, node, "A_Color"), value(context, node, "B_Color"));
        return flag(node, "clamp_result") ? o.saturate(blended) : blended;
    }
    unsupported(node, std::format("mixes {} values", type));
    return zero(identifier == "Result_Float" ? "VALUE" : "VECTOR");
}

GraphCompiler::Value GraphCompiler::mixRgb(const Context& context, NodeRef node) {
    Ops o{builder};
    if (flag(node, "use_alpha"))
        unsupported(node, "mixes by the alpha of its second colour");
    const Value blended =
        blend(text(node, "blend_type"), o.saturate(value(context, node, "Fac")), value(context, node, "Color1"), value(context, node, "Color2"));
    return flag(node, "use_clamp") ? o.saturate(blended) : blended;
}

GraphCompiler::Value GraphCompiler::mapRange(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    const bool vector = text(node, "data_type") == "FLOAT_VECTOR";
    if (vector != (identifier == "Vector"))
        return zero(vector ? "VALUE" : "VECTOR");
    const auto in = [&](const char* scalarName, const char* vectorName) { return value(context, node, vector ? vectorName : scalarName); };
    const Value x = in("Value", "Vector");
    const Value fromMin = in("From Min", "From_Min_FLOAT3");
    const Value fromMax = in("From Max", "From_Max_FLOAT3");
    const Value toMin = in("To Min", "To_Min_FLOAT3");
    const Value toMax = in("To Max", "To_Max_FLOAT3");
    const std::string interpolation = text(node, "interpolation_type");
    Value factor = o.div(o.sub(x, fromMin), o.sub(fromMax, fromMin));
    if (interpolation == "STEPPED") {
        const Value steps = in("Steps", "Steps_FLOAT3");
        factor = o.select(o.greater(steps, o.c(0.0f)), o.div(o.floor(o.mul(factor, o.add(steps, o.c(1.0f)))), steps), o.c(0.0f));
    } else if (interpolation == "SMOOTHSTEP") {
        const Value t = o.saturate(factor);
        factor = o.mul(o.mul(t, t), o.sub(o.c(3.0f), o.mul(t, o.c(2.0f))));
    } else if (interpolation == "SMOOTHERSTEP") {
        const Value t = o.saturate(factor);
        factor = o.mul(o.mul(o.mul(t, t), t), o.add(o.mul(t, o.sub(o.mul(t, o.c(6.0f)), o.c(15.0f))), o.c(10.0f)));
    }
    const Value mapped = o.mulAdd(factor, o.sub(toMax, toMin), toMin);
    const bool clamps = flag(node, "clamp") && (interpolation == "LINEAR" || interpolation == "STEPPED");
    return clamps ? builder.op(GraphOp::Clamp, mapped, toMin, toMax) : mapped;
}

GraphCompiler::Value GraphCompiler::colorRamp(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    const std::string interpolation = text(node, "color_ramp.interpolation");
    const Value f = value(context, node, "Fac");
    const std::span<const float> elements = numbers(node, "color_ramp.elements");
    if (elements.size() < rampElementSize)
        return zero(identifier == "Alpha" ? "VALUE" : "RGBA");
    //! Blender blends hues whatever the interpolation.
    if (text(node, "color_ramp.color_mode") != "RGB" || interpolation == "B_SPLINE" || interpolation == "CARDINAL")
        return sampledRamp(node, elements, f, identifier);
    const auto color = [&](const std::size_t index) {
        const std::span<const float> element = elements.subspan(index * rampElementSize + 1, 4);
        return parameter(std::format("{}.color{}", nodeName(node), index), NodeValues{{element[0], element[1], element[2], element[3]}, 4}, 4);
    };
    Value result = color(0);
    float previous = elements[0];
    for (std::size_t index = 1; index < elements.size() / rampElementSize; ++index) {
        const float position = elements[index * rampElementSize];
        const Value next = color(index);
        if (interpolation == "CONSTANT") {
            result = o.select(o.less(f, o.c(position)), result, next);
        } else {
            const float span = position - previous;
            Value t = span > 0.0f ? o.saturate(o.mul(o.sub(f, o.c(previous)), o.c(1.0f / span))) : o.select(o.less(f, o.c(position)), o.c(0.0f), o.c(1.0f));
            if (interpolation == "EASE")
                t = o.mul(o.mul(t, t), o.sub(o.c(3.0f), o.mul(t, o.c(2.0f))));
            result = o.mix(result, next, t);
        }
        previous = position;
    }
    return identifier == "Alpha" ? o.x(result, 3) : o.rgb(result);
}

//! A spline or hue ramp as linear pieces; Blender blends a spline from 0 and to 1 past the first and last element.
GraphCompiler::Value GraphCompiler::sampledRamp(NodeRef node, const std::span<const float> elements, const Value f, const std::string& identifier) {
    Ops o{builder};
    const RampPieces pieces = rampPieces(elements);
    const std::span<const float> samples = numbers(node, "color_ramp.samples");
    std::size_t next = 0;
    const auto color = [&](const float position) {
        NodeValues values;
        values.count = 4;
        for (std::size_t index = 0; index < 4 && next * 4 + index < samples.size(); ++index)
            values.values[index] = samples[next * 4 + index];
        ++next;
        return parameter(std::format("{}.ramp{}", nodeName(node), position), values, 4);
    };
    Value result = color(pieces.first);
    for (const RampPiece& piece : pieces.pieces)
        result = o.mix(result, color(piece.start + piece.step), o.saturate(o.mul(o.sub(f, o.c(piece.start)), o.c(1.0f / piece.step))));
    return identifier == "Alpha" ? o.x(result, 3) : o.rgb(result);
}

GraphCompiler::Value GraphCompiler::sampledCurve(const std::span<const float> samples, const bool extrapolates, const Value x, const float low,
                                                 const float high) {
    Ops o{builder};
    const auto evaluate = [&](const std::size_t index) { return index < samples.size() ? samples[index] : 0.0f; };
    if (high - low < 1e-6f)
        return o.c(evaluate(0));
    const float step = (high - low) / static_cast<float>(curveSamples - 1);
    const Value t = o.mul(o.sub(x, o.c(low)), o.c(1.0f / step));
    Value result = o.c(evaluate(0));
    for (std::size_t segment = 0; segment + 1 < curveSamples; ++segment) {
        const Value local = o.sub(t, o.c(static_cast<float>(segment)));
        const bool first = segment == 0;
        const bool last = segment + 2 == curveSamples;
        Value weight = o.saturate(local);
        if (extrapolates && first && last)
            weight = local;
        else if (extrapolates && first)
            weight = o.min(local, o.c(1.0f));
        else if (extrapolates && last)
            weight = o.max(local, o.c(0.0f));
        result = o.mix(result, o.c(evaluate(segment + 1)), weight);
    }
    return result;
}

GraphCompiler::Value GraphCompiler::curve(const Context& context, NodeRef node) {
    Ops o{builder};
    const std::string_view kind = nodeType(node);
    const std::size_t curves = static_cast<std::size_t>(integer(node, "mapping.curve_count"));
    const bool extrapolates = text(node, "mapping.extend") == "EXTRAPOLATED";
    const auto points = [&](const std::size_t curve) { return numbers(node, std::format("mapping.curve{}", curve)); };
    const auto samples = [&](const std::size_t channel) { return numbers(node, std::format("mapping.samples{}", channel)); };
    if (kind == "CURVE_FLOAT") {
        if (curves == 0)
            return value(context, node, "Value");
        const Value x = value(context, node, "Value");
        const auto [low, high] = curveRange({points(0)});
        return o.mix(x, sampledCurve(samples(0), extrapolates, x, low, high), value(context, node, "Factor"));
    }
    const bool rgb = kind == "CURVE_RGB";
    const Value x = value(context, node, rgb ? "Color" : "Vector");
    if (curves < (rgb ? 4u : 3u)) {
        unsupported(node, "has too few curves");
        return x;
    }
    std::array<Value, 3> channels{};
    for (std::uint32_t channel = 0; channel < 3; ++channel) {
        const auto [low, high] = rgb ? curveRange({points(3), points(channel)}) : curveRange({points(channel)});
        channels[channel] = sampledCurve(samples(channel), extrapolates, o.x(x, channel), low, high);
    }
    return o.mix(x, o.combine({channels[0], channels[1], channels[2]}), value(context, node, "Fac"));
}

GraphCompiler::Value GraphCompiler::separateColor(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    Value color = value(context, node, "Color");
    const std::string mode = text(node, "mode");
    if (mode == "HSL")
        unsupported(node, "separates HSL; HSV stands in");
    if (mode != "RGB")
        color = o.call(GraphFunction::RgbToHsv, {color});
    return o.x(color, identifier == "Red" ? 0 : identifier == "Green" ? 1 : 2);
}

GraphCompiler::Value GraphCompiler::combineColor(const Context& context, NodeRef node) {
    Ops o{builder};
    const Value color = o.combine({value(context, node, "Red"), value(context, node, "Green"), value(context, node, "Blue")});
    const std::string mode = text(node, "mode");
    if (mode == "HSL")
        unsupported(node, "combines HSL; HSV stands in");
    return mode == "RGB" ? color : o.call(GraphFunction::HsvToRgb, {color});
}

GraphCompiler::Value GraphCompiler::rotate(const Value vector, const Value euler, const bool inverse) {
    Ops o{builder};
    const Value cx = o.cos(o.x(euler, 0));
    const Value sx = o.sin(o.x(euler, 0));
    const Value cy = o.cos(o.x(euler, 1));
    const Value sy = o.sin(o.x(euler, 1));
    const Value cz = o.cos(o.x(euler, 2));
    const Value sz = o.sin(o.x(euler, 2));
    const Value column0 = o.combine({o.mul(cy, cz), o.mul(cy, sz), o.neg(sy)});
    const Value column1 = o.combine({o.sub(o.mul(o.mul(sy, sx), cz), o.mul(cx, sz)), o.add(o.mul(o.mul(sy, sx), sz), o.mul(cx, cz)), o.mul(cy, sx)});
    const Value column2 = o.combine({o.add(o.mul(o.mul(sy, cx), cz), o.mul(sx, sz)), o.sub(o.mul(o.mul(sy, cx), sz), o.mul(sx, cz)), o.mul(cy, cx)});
    if (inverse)
        return o.combine({o.dot(column0, vector), o.dot(column1, vector), o.dot(column2, vector)});
    return o.mulAdd(column0, o.x(vector, 0), o.mulAdd(column1, o.x(vector, 1), o.mul(column2, o.x(vector, 2))));
}

GraphCompiler::Value GraphCompiler::mapping(const Context& context, NodeRef node) {
    Ops o{builder};
    const std::string type = text(node, "vector_type");
    const Value vector = value(context, node, "Vector");
    const Value rotation = value(context, node, "Rotation");
    const Value scale = value(context, node, "Scale");
    if (type == "TEXTURE")
        return o.div(rotate(o.sub(vector, value(context, node, "Location")), rotation, true), scale);
    if (type == "VECTOR")
        return rotate(o.mul(vector, scale), rotation, false);
    if (type == "NORMAL")
        return o.normalize(rotate(o.div(vector, scale), rotation, false));
    return o.add(rotate(o.mul(vector, scale), rotation, false), value(context, node, "Location"));
}

GraphCompiler::Value GraphCompiler::vectorRotate(const Context& context, NodeRef node) {
    Ops o{builder};
    const std::string type = text(node, "rotation_type");
    const bool invert = flag(node, "invert");
    const Value center = value(context, node, "Center");
    const Value local = o.sub(value(context, node, "Vector"), center);
    if (type == "EULER_XYZ")
        return o.add(rotate(local, value(context, node, "Rotation"), invert), center);
    Value angle = value(context, node, "Angle");
    if (invert)
        angle = o.neg(angle);
    Value axis = o.c(Core::float3{0.0f, 0.0f, 1.0f});
    if (type == "X_AXIS")
        axis = o.c(Core::float3{1.0f, 0.0f, 0.0f});
    else if (type == "Y_AXIS")
        axis = o.c(Core::float3{0.0f, 1.0f, 0.0f});
    else if (type == "AXIS_ANGLE")
        axis = value(context, node, "Axis");
    const Value k = o.normalize(axis);
    const Value cosine = o.cos(angle);
    const Value sine = o.sin(angle);
    const Value rotated = o.add(o.add(o.mul(local, cosine), o.mul(o.cross(k, local), sine)), o.mul(k, o.mul(o.dot(k, local), o.sub(o.c(1.0f), cosine))));
    const Value result = o.add(rotated, center);
    if (type != "AXIS_ANGLE")
        return result;
    return o.select(o.greater(builder.op(GraphOp::Length, axis), o.c(0.0f)), result, o.add(local, center));
}

GraphCompiler::Value GraphCompiler::normalInput(const Context& context, NodeRef node, const std::string_view identifier) {
    const SocketRef socket = input(node, identifier);
    if (!socket || !links.linkedNode(socket).node)
        return shadingNormal(context);
    Ops o{builder};
    return o.normalize(socketValue(context, node, socket));
}

GraphCompiler::Value GraphCompiler::normalMap(const Context& context, NodeRef node) {
    Ops o{builder};
    const std::string space = text(node, "space");
    const Value strength = o.max(value(context, node, "Strength"), o.c(0.0f));
    const Value normal = hit(context, GraphInput::Normal);
    Value encoded = o.sub(o.mul(value(context, node, "Color"), o.c(2.0f)), o.c(1.0f));
    if (text(node, "convention") == "DIRECTX")
        encoded = o.mul(encoded, o.c(Core::float3{1.0f, -1.0f, 1.0f}));
    Value bent = o.normalize(encoded);
    if (space == "TANGENT") {
        const std::string map = text(node, "uv_map");
        if (!map.empty() && !result.tangentMap.empty() && result.tangentMap != map)
            diagnostics.warn(std::format("material '{}': the normal map '{}' uses the tangents of '{}', not of '{}'", materialName, nodeName(node),
                                         result.tangentMap, map));
        else
            result.tangentMap = map;
        result.tangents = true;
        const Value backfacing = hit(context, GraphInput::Backfacing);
        const Value frameNormal = o.select(backfacing, o.neg(normal), normal);
        const Value tangentSign = hit(context, GraphInput::Tangent);
        const Value tangent = o.rgb(tangentSign);
        const Value bitangent = o.mul(o.cross(frameNormal, tangent), o.x(tangentSign, 3));
        const Value mapped = o.normalize(o.mulAdd(tangent, o.x(encoded, 0), o.mulAdd(bitangent, o.x(encoded, 1), o.mul(frameNormal, o.x(encoded, 2)))));
        //! A mesh without a UV map has a tangent sign of 0, and Cycles then keeps the normal.
        const Value hasTangent = o.greater(o.abs(o.x(tangentSign, 3)), o.c(0.0f));
        bent = o.select(hasTangent, o.select(backfacing, o.neg(mapped), mapped), normal);
    } else if (space != "WORLD") {
        unsupported(node, std::format("maps normals in the {} space; world space stands in", space));
    }
    return o.normalize(o.mulAdd(o.sub(bent, normal), strength, normal));
}

GraphCompiler::Value GraphCompiler::bump(const Context& context, NodeRef node) {
    Ops o{builder};
    const Value normal = normalInput(context, node, "Normal");
    const SocketRef height = input(node, "Height");
    if (!height || !links.linkedNode(height).node)
        return normal;
    const Value strength = o.saturate(value(context, node, "Strength"));
    Value distance = value(context, node, "Distance");
    if (flag(node, "invert"))
        distance = o.neg(distance);
    const SocketRef filter = inputNamed(node, "Filter Width");
    const Value width = filter ? o.max(value(context, node, "Filter Width"), o.c(1e-5f)) : o.c(1.0f);
    const Value center = socketValue(context, node, height);
    const Value heightX = socketValue({context.frame, {1, width}}, node, height);
    const Value heightY = socketValue({context.frame, {2, width}}, node, height);
    return bumpedNormal(normal, center, heightX, heightY, width, distance, strength);
}

//! Cycles' svm_node_set_bump: the heights at the hit and at the hit moved by width times dPdx and dPdy tilt the normal.
GraphCompiler::Value GraphCompiler::bumpedNormal(const Value normal, const Value center, const Value heightX, const Value heightY, const Value width,
                                                 const Value distance, const Value strength) {
    Ops o{builder};
    const Value dPdx = o.mul(hit({}, GraphInput::DPdx), width);
    const Value dPdy = o.mul(hit({}, GraphInput::DPdy), width);
    const Value rx = o.cross(dPdy, normal);
    const Value ry = o.cross(normal, dPdx);
    const Value determinant = o.dot(dPdx, rx);
    const Value gradient = o.mulAdd(o.sub(heightX, center), rx, o.mul(o.sub(heightY, center), ry));
    const Value bumped = o.normalize(o.sub(o.mul(o.abs(determinant), normal), o.mul(o.mul(distance, builder.op(GraphOp::Sign, determinant)), gradient)));
    return o.normalize(o.mix(normal, bumped, strength));
}

//! The normal of a closure input without a link: the normal of the hit, or the normal that the bump of the displacement gives.
GraphCompiler::Value GraphCompiler::shadingNormal(const Context& context) {
    if (displacedNormal && context.bump.axis == 0)
        return *displacedNormal;
    return hit(context, GraphInput::Normal);
}

//! Cycles' bump_from_displacement: the displacement along the normal is the height of a Bump node with distance 1 and its default filter width.
void GraphCompiler::bumpDisplacement(NodeRef output) {
    const SocketRef socket = input(output, "Displacement");
    if (!socket || !links.linkedNode(socket).node)
        return;
    Ops o{builder};
    constexpr float filterWidth = 0.1f;
    const Value width = o.c(filterWidth);
    const Value normal = hit({}, GraphInput::Normal);
    const auto height = [&](const Bump bump) { return o.dot(socketValue({0, bump}, output, socket), normal); };
    displacedNormal = bumpedNormal(normal, height({}), height({1, width}), height({2, width}), width, o.c(1.0f), o.c(1.0f));
}

//! Cycles' svm_node_displacement: the height along the normal, in object space when the node says so.
GraphCompiler::Value GraphCompiler::displacement(const Context& context, NodeRef node) {
    Ops o{builder};
    const Value amount = o.mul(o.sub(value(context, node, "Height"), value(context, node, "Midlevel")), value(context, node, "Scale"));
    Value normal = value(context, node, "Normal");
    if (text(node, "space") != "OBJECT")
        return o.mul(normal, amount);
    const std::array<Value, 3> toWorld = rows(context, GraphInput::ObjectToWorld0);
    normal = o.normalize(o.mulAdd(o.rgb(toWorld[0]), o.x(normal, 0), o.mulAdd(o.rgb(toWorld[1]), o.x(normal, 1), o.mul(o.rgb(toWorld[2]), o.x(normal, 2)))));
    return o.transform(toWorld, o.mul(normal, amount), false);
}

//! Cycles' svm_node_vector_displacement: a tangent space vector uses the object space tangent frame of the render UV map.
GraphCompiler::Value GraphCompiler::vectorDisplacement(const Context& context, NodeRef node) {
    Ops o{builder};
    const std::string space = text(node, "space");
    Value offset = o.mul(o.sub(value(context, node, "Vector"), value(context, node, "Midlevel")), value(context, node, "Scale"));
    if (space == "WORLD")
        return offset;
    const std::array<Value, 3> toWorld = rows(context, GraphInput::ObjectToWorld0);
    if (space == "TANGENT") {
        result.tangents = true;
        const Value worldNormal = hit(context, GraphInput::Normal);
        const Value normal = o.normalize(
            o.mulAdd(o.rgb(toWorld[0]), o.x(worldNormal, 0), o.mulAdd(o.rgb(toWorld[1]), o.x(worldNormal, 1), o.mul(o.rgb(toWorld[2]), o.x(worldNormal, 2)))));
        const Value tangentSign = hit(context, GraphInput::Tangent);
        const Value tangent = o.normalize(o.transform(rows(context, GraphInput::WorldToObject0), o.rgb(tangentSign), false));
        const Value bitangent = o.mul(o.cross(normal, tangent), o.x(tangentSign, 3));
        offset = o.mulAdd(tangent, o.x(offset, 0), o.mulAdd(normal, o.x(offset, 1), o.mul(bitangent, o.x(offset, 2))));
    }
    return o.transform(toWorld, offset, false);
}

GraphCompiler::Value GraphCompiler::gabor(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    const auto in = [&](const std::string_view name) { return value(context, node, name); };
    const Value settings = o.combine({in("Scale"), in("Frequency"), in("Anisotropy"), in("Orientation 2D")});
    const float dimensions = text(node, "gabor_type") == "2D" ? 2.0f : 3.0f;
    const Value result = o.call(GraphFunction::Gabor, {textureVector(context, node), settings, in("Orientation 3D"), o.c(dimensions)});
    return o.x(result, identifier == "Value" ? 0 : identifier == "Phase" ? 1 : 2);
}

//! Cycles' svm_node_ies: the vector defaults to the incoming direction in object space, and the angles look the profile table up.
GraphCompiler::Value GraphCompiler::ies(const Context& context, NodeRef node) {
    Ops o{builder};
    const std::expected<NodeTexture, std::string> table = textures.load(*node.tree, node.index, NodeTextureUse::Ies);
    if (!table) {
        unsupported(node, table.error());
        return o.c(0.0f);
    }
    Scene::MaterialTexture texture;
    texture.texture = table->texture;
    texture.wrap = Scene::TextureWrap::Extend;
    Value vector = o.rgb(hit(context, GraphInput::Incoming));
    vector = linked(node, "Vector") ? value(context, node, "Vector") : o.transposed(rows(context, GraphInput::ObjectToWorld0), vector);
    const Value direction = o.normalize(vector);
    constexpr float pi = std::numbers::pi_v<float>;
    const Value vertical = builder.op(GraphOp::Acos, builder.op(GraphOp::Clamp, o.neg(o.x(direction, 2)), o.c(-1.0f), o.c(1.0f)));
    const Value horizontal = o.add(builder.op(GraphOp::Atan2, o.x(direction, 0), o.x(direction, 1)), o.c(pi));
    const Value lookup = builder.texture(textureSlot(texture), o.combine({o.div(horizontal, o.c(2.0f * pi)), o.div(vertical, o.c(pi))}), o.c(Core::float4{}));
    return o.mul(value(context, node, "Strength"), o.x(lookup, 0));
}

//! Cycles looks the sky up in the direction of the vector without normalizing it, as the table of the world does.
GraphCompiler::Value GraphCompiler::sky(const Context& context, NodeRef node) {
    Ops o{builder};
    const std::expected<NodeTexture, std::string> table = textures.load(*node.tree, node.index, NodeTextureUse::Sky);
    if (!table)
        return zero("RGBA");
    Scene::MaterialTexture texture;
    texture.texture = table->texture;
    texture.wrap = Scene::TextureWrap::Repeat;
    const Value vector = linked(node, "Vector") ? value(context, node, "Vector") : hit(context, GraphInput::Generated);
    constexpr float pi = std::numbers::pi_v<float>;
    const Value u = o.sub(o.c(0.5f), o.div(builder.op(GraphOp::Atan2, o.x(vector, 1), o.x(vector, 0)), o.c(2.0f * pi)));
    const Value v = o.sub(o.c(1.0f), o.div(builder.op(GraphOp::Acos, builder.op(GraphOp::Clamp, o.x(vector, 2), o.c(-1.0f), o.c(1.0f))), o.c(pi)));
    return o.rgb(builder.texture(textureSlot(texture), o.combine({u, v}), o.c(Core::float4{})));
}

GraphCompiler::Value GraphCompiler::radialTiling(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    const Value vector = value(context, node, "Vector");
    if (identifier == "Segment ID")
        return o.call(GraphFunction::RadialTilingSegment, {vector, value(context, node, "Sides")});
    const Value settings =
        o.combine({value(context, node, "Sides"), value(context, node, "Roundness"), o.c(flag(node, "normalize") ? 1.0f : 0.0f), o.c(0.0f)});
    const Value result = o.call(GraphFunction::RadialTiling, {vector, settings});
    if (identifier == "Segment Coordinates")
        return o.combine({o.x(result, 0), o.x(result, 1), o.c(0.0f)});
    return o.x(result, identifier == "Segment Width" ? 2 : 3);
}

GraphCompiler::Value GraphCompiler::textureVector(const Context& context, NodeRef node) {
    const SocketRef socket = input(node, "Vector");
    if (!socket || !links.linkedNode(socket).node)
        return hit(context, GraphInput::Generated);
    return socketValue(context, node, socket);
}

GraphCompiler::Value GraphCompiler::coordinate(const Context& context, NodeRef node, const std::string_view dimensions) {
    Ops o{builder};
    const Value vector = dimensions == "1D" ? o.c(Core::float3{}) : textureVector(context, node);
    const Value w = dimensions == "1D" || dimensions == "4D" ? value(context, node, "W") : o.c(0.0f);
    return o.combine({vector, w});
}

std::optional<GraphCompiler::ImageSlot> GraphCompiler::imageSlot(NodeRef node, const Scene::TextureWrap wrap) {
    if (text(node, "image").empty())
        return std::nullopt;
    const bool readsAlpha = links.isLinked(outputNamed(node, "Alpha"));
    const std::expected<NodeTexture, std::string> image =
        textures.load(*node.tree, node.index, readsAlpha ? NodeTextureUse::Straight : NodeTextureUse::Premultiplied);
    if (!image)
        return std::nullopt;
    Scene::MaterialTexture texture;
    texture.texture = image->texture;
    texture.wrap = text(node, "image_source") == "TILED" ? Scene::TextureWrap::Clip : wrap;
    texture.filter = toFilter(text(node, "interpolation"));
    return ImageSlot{textureSlot(texture), image->columns, image->rows};
}

//! Cycles shows a texture node without an image in magenta.
GraphCompiler::Value GraphCompiler::missingImage(const std::string& identifier) {
    Ops o{builder};
    return identifier == "Alpha" ? o.c(1.0f) : o.c(Core::float3{1.0f, 0.0f, 1.0f});
}

//! Cycles looks the image up in the direction of the vector, which is the hit position when the socket has no link.
GraphCompiler::Value GraphCompiler::environment(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    const std::optional<ImageSlot> slot = imageSlot(node, Scene::TextureWrap::Repeat);
    if (!slot)
        return missingImage(identifier);
    const SocketRef socket = input(node, "Vector");
    const bool linkedVector = socket && links.linkedNode(socket).node;
    const bool mirrorBall = text(node, "projection") == "MIRROR_BALL";
    const auto coordinate = [&](const Context& at) {
        const Value direction = o.normalize(linkedVector ? socketValue(at, node, socket) : hit(at, GraphInput::Position));
        const Value dx = o.x(direction, 0);
        const Value dy = o.x(direction, 1);
        const Value dz = o.x(direction, 2);
        if (mirrorBall) {
            const Value y = o.sub(dy, o.c(1.0f));
            const Value divisor = o.mul(o.c(2.0f), builder.op(GraphOp::Sqrt, o.max(o.mul(y, o.c(-0.5f)), o.c(0.0f))));
            const Value scale = o.div(o.c(0.5f), o.max(divisor, o.c(1e-20f)));
            return o.combine({o.add(o.mul(dx, scale), o.c(0.5f)), o.add(o.mul(dz, scale), o.c(0.5f))});
        }
        const Value u = o.sub(o.c(0.5f), o.div(builder.op(GraphOp::Atan2, dy, dx), o.c(2.0f * std::numbers::pi_v<float>)));
        const Value v =
            o.sub(o.c(1.0f), o.div(builder.op(GraphOp::Acos, builder.op(GraphOp::Clamp, dz, o.c(-1.0f), o.c(1.0f))), o.c(std::numbers::pi_v<float>)));
        return o.combine({u, v});
    };
    const Value texel = sample(context, slot->slot, coordinate, !mirrorBall);
    return identifier == "Alpha" ? o.x(texel, 3) : o.rgb(texel);
}

GraphCompiler::Value GraphCompiler::image(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    const std::optional<ImageSlot> found = imageSlot(node, toWrap(text(node, "extension")).value_or(Scene::TextureWrap::Repeat));
    if (!found)
        return missingImage(identifier);
    const std::uint32_t slot = found->slot;
    const Core::float2 tiles{1.0f / float(found->columns), 1.0f / float(found->rows)};
    const SocketRef vectorSocket = input(node, "Vector");
    const bool linkedVector = vectorSocket && links.linkedNode(vectorSocket).node;
    const auto vector = [&](const Context& at) { return linkedVector ? socketValue(at, node, vectorSocket) : uv(at, ""); };
    const std::string projection = text(node, "projection");
    Value texel;
    if (projection == "SPHERE" || projection == "TUBE") {
        const auto coordinate = [&](const Context& at) {
            const Value p = o.sub(o.mul(vector(at), o.c(2.0f)), o.c(1.0f));
            const Value px = o.x(p, 0);
            const Value py = o.x(p, 1);
            const Value pz = o.x(p, 2);
            const Value u = o.mul(o.sub(o.c(1.0f), o.div(builder.op(GraphOp::Atan2, px, py), o.c(std::numbers::pi_v<float>))), o.c(0.5f));
            const Value v = projection == "TUBE"
                                ? o.mul(o.add(pz, o.c(1.0f)), o.c(0.5f))
                                : o.sub(o.c(1.0f), o.div(builder.op(GraphOp::Acos, o.div(pz, builder.op(GraphOp::Length, p))), o.c(std::numbers::pi_v<float>)));
            return o.combine({u, v});
        };
        texel = sample(context, slot, coordinate, true);
    } else if (projection == "BOX") {
        texel = boxTexture(context, slot, vector, number(node, "projection_blend"));
    } else {
        texel = sample(context, slot, [&](const Context& at) { return tiles.x == 1.0f && tiles.y == 1.0f ? builder.resize(vector(at), 2) : o.mul(builder.resize(vector(at), 2), o.combine({o.c(tiles.x), o.c(tiles.y)})); }, false);
    }
    return identifier == "Alpha" ? o.x(texel, 3) : o.rgb(texel);
}

//! Cycles' svm_node_tex_image_box: three planar lookups, weighted by the object space normal, and mapped so that no side is mirrored.
GraphCompiler::Value GraphCompiler::boxTexture(const Context& context, const std::uint32_t slot, const Coordinate& vector, const float blend) {
    Ops o{builder};
    const Value normal = o.transposed(rows(context, GraphInput::ObjectToWorld0), shadingNormal(context));
    const Value weights = o.call(GraphFunction::BoxWeights, {normal, o.c(blend)});
    const auto face = [&](const std::uint32_t axis, const bool positive, const std::uint32_t flippedComponent, const std::uint32_t component) {
        const Value side = positive ? o.greater(o.x(normal, axis), o.c(0.0f)) : o.less(o.x(normal, axis), o.c(0.0f));
        return sample(
            context, slot,
            [&](const Context& at) {
                const Value point = vector(at);
                const Value flipped = o.x(point, flippedComponent);
                return o.combine({o.select(side, o.sub(o.c(1.0f), flipped), flipped), o.x(point, component)});
            },
            false);
    };
    const Value x = face(0, false, 1, 2);
    const Value y = face(1, true, 0, 2);
    const Value z = face(2, true, 1, 0);
    return o.mulAdd(x, o.x(weights, 0), o.mulAdd(y, o.x(weights, 1), o.mul(z, o.x(weights, 2))));
}

GraphCompiler::Value GraphCompiler::sample(const Context& context, const std::uint32_t slot, const Coordinate& coordinate, const bool wraps) {
    Ops o{builder};
    const Value one = o.c(1.0f);
    const Value uv = coordinate(context);
    const Value center = context.bump.axis == 0 ? uv : coordinate({context.frame, {}});
    const auto change = [&](const std::uint8_t axis) {
        const Value delta = o.sub(coordinate({context.frame, {axis, one}}), center);
        if (!wraps)
            return delta;
        const Value u = o.x(delta, 0);
        return o.combine({o.sub(u, builder.op(GraphOp::Round, u)), o.x(delta, 1)});
    };
    return builder.texture(slot, uv, o.combine({change(1), change(2)}));
}

GraphCompiler::Value GraphCompiler::noise(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    static constexpr std::array<std::string_view, 5> types = {"FBM", "MULTIFRACTAL", "HYBRID_MULTIFRACTAL", "RIDGED_MULTIFRACTAL", "HETERO_TERRAIN"};
    const std::string dimensions = text(node, "noise_dimensions");
    const std::string type = text(node, "noise_type");
    const auto in = [&](const std::string_view name) { return value(context, node, name); };
    const bool offset = type != "FBM" && type != "MULTIFRACTAL";
    const bool gain = type == "HYBRID_MULTIFRACTAL" || type == "RIDGED_MULTIFRACTAL";
    const Value settings = o.combine({in("Scale"), in("Detail"), in("Roughness"), in("Lacunarity")});
    const Value options = o.combine(
        {in("Distortion"), o.c(flag(node, "normalize") ? 1.0f : 0.0f), o.c(static_cast<float>(dimensions[0] - '0')), o.c(indexOf(types, type))});
    const Value shape = o.combine({offset ? in("Offset") : o.c(0.0f), gain ? in("Gain") : o.c(1.0f), o.c(0.0f), o.c(0.0f)});
    const Value result = o.call(GraphFunction::Noise, {coordinate(context, node, dimensions), settings, options, shape});
    return identifier == "Fac" ? o.x(result, 3) : o.rgb(result);
}

GraphCompiler::Value GraphCompiler::voronoi(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    static constexpr std::array<std::string_view, 4> metrics = {"EUCLIDEAN", "MANHATTAN", "CHEBYCHEV", "MINKOWSKI"};
    static constexpr std::array<std::string_view, 5> features = {"F1", "F2", "SMOOTH_F1", "DISTANCE_TO_EDGE", "N_SPHERE_RADIUS"};
    const std::string dimensions = text(node, "voronoi_dimensions");
    const std::string metric = text(node, "distance");
    const std::string feature = text(node, "feature");
    const auto in = [&](const std::string_view name) { return value(context, node, name); };
    const bool fractal = feature != "N_SPHERE_RADIUS";
    const Value settings =
        o.combine({in("Scale"), fractal ? in("Detail") : o.c(0.0f), fractal ? in("Roughness") : o.c(0.5f), fractal ? in("Lacunarity") : o.c(2.0f)});
    const Value options = o.combine({in("Randomness"), metric == "MINKOWSKI" ? in("Exponent") : o.c(0.5f),
                                     feature == "SMOOTH_F1" ? in("Smoothness") : o.c(1.0f), o.c(flag(node, "normalize") ? 1.0f : 0.0f)});
    const Value kind = o.combine({o.c(indexOf(metrics, metric)), o.c(indexOf(features, feature)), o.c(static_cast<float>(dimensions[0] - '0')), o.c(0.0f)});
    const Value point = coordinate(context, node, dimensions);
    if (identifier == "Position" || identifier == "W") {
        const Value position = o.call(GraphFunction::VoronoiPosition, {point, settings, options, kind});
        return identifier == "W" ? o.x(position, 3) : o.rgb(position);
    }
    const Value result = o.call(GraphFunction::Voronoi, {point, settings, options, kind});
    return identifier == "Color" ? o.rgb(result) : o.x(result, 3);
}

GraphCompiler::Value GraphCompiler::wave(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    static constexpr std::array<std::string_view, 4> bands = {"X", "Y", "Z", "DIAGONAL"};
    static constexpr std::array<std::string_view, 4> rings = {"X", "Y", "Z", "SPHERICAL"};
    static constexpr std::array<std::string_view, 3> profiles = {"SIN", "SAW", "TRI"};
    const bool ringed = text(node, "wave_type") == "RINGS";
    const float direction = ringed ? indexOf(rings, text(node, "rings_direction")) : indexOf(bands, text(node, "bands_direction"));
    const auto in = [&](const std::string_view name) { return value(context, node, name); };
    const Value settings = o.combine({in("Scale"), in("Distortion"), in("Detail"), in("Detail Scale")});
    const Value options = o.combine({in("Detail Roughness"), in("Phase Offset"), o.c(ringed ? 1.0f : 0.0f), o.c(direction)});
    const Value result =
        o.call(GraphFunction::Wave, {textureVector(context, node), settings, options, o.c(indexOf(profiles, text(node, "wave_profile")))});
    return identifier == "Fac" ? result : builder.resize(result, 3);
}

GraphCompiler::Value GraphCompiler::magic(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    const Value settings = o.combine(
        {value(context, node, "Scale"), value(context, node, "Distortion"), o.c(static_cast<float>(integer(node, "turbulence_depth"))), o.c(0.0f)});
    const Value result = o.call(GraphFunction::Magic, {textureVector(context, node), settings});
    return identifier == "Fac" ? o.x(result, 3) : o.rgb(result);
}

GraphCompiler::Value GraphCompiler::checker(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    const Value factor = o.call(GraphFunction::Checker, {textureVector(context, node), value(context, node, "Scale")});
    if (identifier == "Fac")
        return factor;
    return o.mix(value(context, node, "Color2"), value(context, node, "Color1"), factor);
}

GraphCompiler::Value GraphCompiler::brick(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    const auto in = [&](const std::string_view name) { return value(context, node, name); };
    const Value offsets = o.combine({o.c(number(node, "offset")), o.c(static_cast<float>(integer(node, "offset_frequency"))),
                                     o.c(number(node, "squash")), o.c(static_cast<float>(integer(node, "squash_frequency")))});
    const Value settings = o.combine({in("Scale"), in("Mortar Size"), in("Mortar Smooth"), in("Bias")});
    const Value result = o.call(GraphFunction::Brick, {textureVector(context, node), offsets, settings, o.combine({in("Brick Width"), in("Row Height")})});
    const Value mortar = o.x(result, 1);
    if (identifier == "Fac")
        return mortar;
    const Value tinted = o.mix(in("Color1"), in("Color2"), o.x(result, 0));
    return o.mix(tinted, in("Mortar"), mortar);
}

GraphCompiler::Value GraphCompiler::gradient(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    static constexpr std::array<std::string_view, 7> types = {"LINEAR", "QUADRATIC", "EASING", "DIAGONAL", "RADIAL", "QUADRATIC_SPHERE", "SPHERICAL"};
    const Value result = o.call(GraphFunction::Gradient, {textureVector(context, node), o.c(indexOf(types, text(node, "gradient_type")))});
    return identifier == "Fac" ? result : builder.resize(result, 3);
}

GraphCompiler::Value GraphCompiler::whiteNoise(const Context& context, NodeRef node, const std::string& identifier) {
    Ops o{builder};
    const std::string dimensions = text(node, "noise_dimensions");
    const Value result = o.call(GraphFunction::WhiteNoise, {coordinate(context, node, dimensions), o.c(static_cast<float>(dimensions[0] - '0'))});
    return identifier == "Value" ? o.x(result, 3) : o.rgb(result);
}

GraphCompiler::Bsdf GraphCompiler::neutralBsdf(const Context& context, const Value weight) {
    Ops o{builder};
    const Value normal = hit(context, GraphInput::Normal);
    return {
        .weight = weight,
        .baseColor = o.c(Core::float3{0.8f, 0.8f, 0.8f}),
        .metallic = o.c(0.0f),
        .roughness = o.c(0.5f),
        .ior = o.c(1.5f),
        .specular = o.c(0.5f),
        .specularTint = o.c(Core::float3{1.0f, 1.0f, 1.0f}),
        .anisotropic = o.c(0.0f),
        .rotation = o.c(0.0f),
        .normal = normal,
        .tangent = o.rgb(hit(context, GraphInput::Tangent)),
        .transmission = o.c(0.0f),
        .coat = o.c(0.0f),
        .coatRoughness = o.c(0.03f),
        .coatIor = o.c(1.5f),
        .coatTint = o.c(Core::float3{1.0f, 1.0f, 1.0f}),
        .coatNormal = normal,
        .sheen = o.c(0.0f),
        .sheenRoughness = o.c(0.5f),
        .sheenTint = o.c(Core::float3{1.0f, 1.0f, 1.0f}),
        .thinFilmThickness = o.c(0.0f),
        .thinFilmIor = o.c(1.33f),
        .mode = o.c(0.0f),
        .diffuseRoughness = o.c(0.0f),
        .subsurface = o.c(0.0f),
        .subsurfaceRadius = o.c(Core::float3{}),
    };
}

void GraphCompiler::scale(Shader& shader, const Value factor) {
    Ops o{builder};
    for (Bsdf& bsdf : shader.bsdfs)
        bsdf.weight = o.mul(bsdf.weight, factor);
    std::erase_if(shader.bsdfs, [&](const Bsdf& bsdf) {
        const float* weight = builder.constantOf(bsdf.weight);
        return weight && std::all_of(weight, weight + builder.width(bsdf.weight), [](const float value) { return value == 0.0f; });
    });
    if (shader.emission)
        shader.emission = o.mul(*shader.emission, factor);
    if (shader.transparency)
        shader.transparency = o.mul(*shader.transparency, factor);
}

void GraphCompiler::add(Shader& target, Shader source) {
    Ops o{builder};
    target.bsdfs.insert(target.bsdfs.end(), source.bsdfs.begin(), source.bsdfs.end());
    if (source.emission)
        target.emission = target.emission ? o.add(*target.emission, *source.emission) : *source.emission;
    if (source.transparency)
        target.transparency = target.transparency ? o.add(*target.transparency, *source.transparency) : *source.transparency;
}

GraphCompiler::Shader GraphCompiler::shaderInput(const Context& context, NodeRef node, const std::string_view identifier) {
    const SocketRef socket = input(node, identifier);
    if (!socket)
        return {};
    const NodeLinkEnd link = links.linkedNode(socket);
    if (!link.node)
        return {};
    const std::string from = socketType(link.socket);
    if (from != "SHADER") {
        Shader emission;
        emission.emission = convert(output(context, link.node, link.socket), from, "RGBA");
        return emission;
    }
    return shader(context, link.node, link.socket);
}

GraphCompiler::Shader GraphCompiler::shader(const Context& context, NodeRef node, SocketRef socket) {
    const std::string identifier = socket->identifier;
    const Key key{context.frame, context.bump, address(node), identifier};
    if (const auto found = shaders.find(key); found != shaders.end())
        return found->second;
    Shader result;
    const std::string_view kind = nodeType(node);
    if (node->muted) {
        if (const SocketRef from = internalLink(node, identifier))
            result = shaderInput(context, node, from->identifier);
    } else if (kind == "GROUP_INPUT") {
        if (context.frame != 0)
            result = shaderInput({frames[context.frame].parent, context.bump}, frames[context.frame].group, identifier);
    } else if (kind == "GROUP") {
        const NodeTree* tree = groupTree(node);
        const NodeRef groupOutput = tree ? findGroupOutput(*tree) : NodeRef{};
        if (groupOutput)
            result = shaderInput({frameOf(context.frame, node), context.bump}, groupOutput, identifier);
        else
            unsupported(node, "is a group without an output");
    } else {
        result = compileShader(context, node);
    }
    shaders.emplace(key, result);
    return result;
}

GraphCompiler::Shader GraphCompiler::principled(const Context& context, NodeRef node, const Value weight) {
    Ops o{builder};
    const auto in = [&](const std::string_view name) { return value(context, node, name); };
    const auto constantInput = [&](const std::string_view name) {
        const SocketRef socket = input(node, name);
        return socket && !links.linkedNode(socket).node ? std::optional(socketValues(socket).values[0]) : std::nullopt;
    };
    Bsdf bsdf = neutralBsdf(context, weight);
    bsdf.baseColor = in("Base Color");
    bsdf.metallic = in("Metallic");
    bsdf.roughness = in("Roughness");
    bsdf.ior = in("IOR");
    bsdf.specular = in("Specular IOR Level");
    bsdf.specularTint = in("Specular Tint");
    bsdf.anisotropic = in("Anisotropic");
    bsdf.rotation = in("Anisotropic Rotation");
    bsdf.normal = normalInput(context, node, "Normal");
    bsdf.tangent = anisotropicTangent(context, node, bsdf.anisotropic);
    bsdf.transmission = in("Transmission Weight");
    bsdf.coat = in("Coat Weight");
    bsdf.coatRoughness = in("Coat Roughness");
    bsdf.coatIor = in("Coat IOR");
    bsdf.coatTint = in("Coat Tint");
    bsdf.coatNormal = normalInput(context, node, "Coat Normal");
    bsdf.sheen = in("Sheen Weight");
    bsdf.sheenRoughness = in("Sheen Roughness");
    bsdf.sheenTint = in("Sheen Tint");
    bsdf.thinFilmThickness = in("Thin Film Thickness");
    bsdf.thinFilmIor = in("Thin Film IOR");
    bsdf.subsurface = o.saturate(in("Subsurface Weight"));
    bsdf.subsurfaceRadius = o.max(o.mul(in("Subsurface Radius"), in("Subsurface Scale")), o.c(Core::float3{}));
    if (constantInput("Subsurface Weight").value_or(1.0f) != 0.0f)
        warnSubsurfaceMethod(node);
    bsdf.diffuseRoughness = o.saturate(in("Diffuse Roughness"));
    if (constantInput("Thin Wall").value_or(1.0f) != 0.0f)
        diagnostics.warn(std::format("material '{}': the thin wall of '{}' is ignored", materialName, nodeName(node)));
    const Value alpha = o.saturate(in("Alpha"));
    Shader shader;
    shader.bsdfs.push_back(bsdf);
    shader.emission = o.mul(in("Emission Color"), in("Emission Strength"));
    shader.transparency = o.mul(o.c(Core::float3{1.0f, 1.0f, 1.0f}), o.sub(o.c(1.0f), alpha));
    for (Bsdf& closure : shader.bsdfs)
        closure.weight = o.mul(closure.weight, alpha);
    *shader.emission = o.mul(*shader.emission, alpha);
    return shader;
}

//! The renderer has only the disk method of Christensen-Burley, the fastest of the Cycles methods.
void GraphCompiler::warnSubsurfaceMethod(NodeRef node) {
    const std::string method = text(node, nodeType(node) == "BSDF_PRINCIPLED" ? "subsurface_method" : "falloff");
    if (method != "BURLEY")
        diagnostics.warn(
            std::format("material '{}': the {} subsurface scattering of '{}' renders as Christensen-Burley", materialName, method, nodeName(node)));
}

GraphCompiler::Shader GraphCompiler::compileShader(const Context& context, NodeRef node) {
    Ops o{builder};
    const std::string_view kind = nodeType(node);
    const auto in = [&](const std::string_view name) { return value(context, node, name); };
    const Value one = o.c(Core::float3{1.0f, 1.0f, 1.0f});
    if (kind == "MIX_SHADER") {
        const Value factor = o.saturate(in("Fac"));
        Shader first = shaderInput(context, node, "Shader");
        Shader second = shaderInput(context, node, "Shader_001");
        scale(first, o.sub(o.c(1.0f), factor));
        scale(second, factor);
        add(first, std::move(second));
        return first;
    }
    if (kind == "ADD_SHADER") {
        Shader first = shaderInput(context, node, "Shader");
        add(first, shaderInput(context, node, "Shader_001"));
        return first;
    }
    if (kind == "BSDF_PRINCIPLED")
        return principled(context, node, one);
    Shader shader;
    Bsdf bsdf = neutralBsdf(context, one);
    const auto leaf = [&] {
        shader.bsdfs.push_back(bsdf);
        return shader;
    };
    if (kind == "EMISSION" || kind == "BACKGROUND") {
        shader.emission = o.mul(in("Color"), in("Strength"));
        return shader;
    }
    if (kind == "BSDF_TRANSPARENT") {
        shader.transparency = in("Color");
        return shader;
    }
    if (kind == "HOLDOUT") {
        bsdf.mode = o.c(2.0f);
        return leaf();
    }
    if (kind == "BSDF_RAY_PORTAL") {
        unsupported(node, "is a ray portal; a transparent closure stands in");
        shader.transparency = in("Color");
        return shader;
    }
    if (kind == "BSDF_DIFFUSE" || kind == "SUBSURFACE_SCATTERING" || kind == "BSDF_TOON" || kind == "BSDF_HAIR" || kind == "BSDF_HAIR_PRINCIPLED") {
        if (kind == "SUBSURFACE_SCATTERING") {
            bsdf.subsurface = o.c(1.0f);
            bsdf.subsurfaceRadius = o.max(o.mul(in("Radius"), in("Scale")), o.c(Core::float3{}));
            warnSubsurfaceMethod(node);
        } else if (kind != "BSDF_DIFFUSE")
            unsupported(node, std::format("({}) has no graph translation; a diffuse closure stands in", kind));
        else
            bsdf.diffuseRoughness = o.saturate(in("Roughness"));
        bsdf.baseColor = in("Color");
        bsdf.specular = o.c(0.0f);
        if (inputNamed(node, "Normal"))
            bsdf.normal = normalInput(context, node, "Normal");
        return leaf();
    }
    if (kind == "BSDF_GLOSSY" || kind == "BSDF_ANISOTROPIC" || kind == "BSDF_METALLIC") {
        const bool metallic = kind == "BSDF_METALLIC";
        bsdf.metallic = o.c(1.0f);
        if (metallic && text(node, "fresnel_type") == "PHYSICAL_CONDUCTOR") {
            const Value n = in("IOR");
            const Value k = in("Extinction");
            const Value kk = o.mul(k, k);
            const Value below = o.sub(n, o.c(1.0f));
            const Value above = o.add(n, o.c(1.0f));
            bsdf.baseColor = o.div(o.add(o.mul(below, below), kk), o.add(o.mul(above, above), kk));
        } else {
            bsdf.baseColor = in(metallic ? "Base Color" : "Color");
            bsdf.specularTint = metallic ? in("Edge Tint") : bsdf.baseColor;
        }
        bsdf.roughness = in("Roughness");
        bsdf.anisotropic = o.max(in("Anisotropy"), o.c(0.0f));
        bsdf.rotation = in("Rotation");
        bsdf.normal = normalInput(context, node, "Normal");
        bsdf.tangent = anisotropicTangent(context, node, bsdf.anisotropic);
        if (metallic) {
            bsdf.thinFilmThickness = in("Thin Film Thickness");
            bsdf.thinFilmIor = in("Thin Film IOR");
        }
        return leaf();
    }
    if (kind == "BSDF_GLASS" || kind == "BSDF_REFRACTION") {
        if (kind == "BSDF_REFRACTION")
            bsdf.mode = o.c(3.0f);
        bsdf.baseColor = in("Color");
        bsdf.roughness = in("Roughness");
        bsdf.ior = in("IOR");
        bsdf.transmission = o.c(1.0f);
        bsdf.normal = normalInput(context, node, "Normal");
        if (kind == "BSDF_GLASS") {
            bsdf.thinFilmThickness = in("Thin Film Thickness");
            bsdf.thinFilmIor = in("Thin Film IOR");
        }
        return leaf();
    }
    if (kind == "BSDF_TRANSLUCENT") {
        bsdf.baseColor = in("Color");
        bsdf.specular = o.c(0.0f);
        bsdf.normal = normalInput(context, node, "Normal");
        bsdf.mode = o.c(1.0f);
        return leaf();
    }
    if (kind == "BSDF_SHEEN") {
        bsdf.baseColor = o.c(Core::float3{});
        bsdf.specular = o.c(0.0f);
        bsdf.sheen = o.c(1.0f);
        bsdf.sheenTint = in("Color");
        bsdf.sheenRoughness = in("Roughness");
        bsdf.normal = normalInput(context, node, "Normal");
        return leaf();
    }
    unsupported(node, std::format("({}) is no surface shader that a material graph knows", kind));
    return shader;
}

void GraphCompiler::finish(Shader shader) {
    Ops o{builder};
    const Value zeroValue = o.c(0.0f);
    const Value one = o.c(1.0f);
    const Value transparency = shader.transparency ? o.saturate(o.dot(*shader.transparency, o.c(averageWeights))) : zeroValue;
    const Value opacity = o.sub(one, transparency);
    if (shader.bsdfs.empty())
        shader.bsdfs.push_back(neutralBsdf({}, o.c(Core::float3{})));
    constexpr std::size_t fieldCount = 15;
    const auto fields = [&](const Bsdf& bsdf, const Value weightOverProbability) {
        return std::array<Value, fieldCount>{
            o.combine({bsdf.baseColor, one}),
            o.combine({bsdf.metallic, bsdf.roughness, bsdf.ior, bsdf.specular}),
            o.combine({bsdf.specularTint, bsdf.anisotropic}),
            o.combine({o.toNoorRay(bsdf.normal), bsdf.rotation}),
            o.combine({o.toNoorRay(bsdf.tangent), bsdf.transmission}),
            o.combine({bsdf.coat, bsdf.coatRoughness, bsdf.coatIor, zeroValue}),
            o.combine({bsdf.coatTint, zeroValue}),
            o.combine({o.toNoorRay(bsdf.coatNormal), zeroValue}),
            o.combine({bsdf.sheen, bsdf.sheenRoughness, bsdf.diffuseRoughness, zeroValue}),
            o.combine({bsdf.sheenTint, zeroValue}),
            zeroValue,
            o.combine({bsdf.thinFilmThickness, bsdf.thinFilmIor, zeroValue, zeroValue}),
            weightOverProbability,
            bsdf.mode,
            o.combine({bsdf.subsurfaceRadius, bsdf.subsurface}),
        };
    };
    std::array<Value, fieldCount> chosen{};
    if (shader.bsdfs.size() == 1) {
        chosen = fields(shader.bsdfs[0], shader.bsdfs[0].weight);
    } else {
        std::vector<Value> luminances;
        Value total = zeroValue;
        for (const Bsdf& bsdf : shader.bsdfs) {
            luminances.push_back(o.max(o.dot(bsdf.weight, o.c(luminanceWeights)), zeroValue));
            total = o.add(total, luminances.back());
        }
        const Value random = hit({}, GraphInput::ClosureRandom);
        std::vector<Value> bounds;
        Value sum = zeroValue;
        for (const Value luminance : luminances) {
            sum = o.add(sum, luminance);
            bounds.push_back(o.div(sum, total));
        }
        const std::size_t last = shader.bsdfs.size() - 1;
        chosen = fields(shader.bsdfs[last], o.mul(shader.bsdfs[last].weight, o.div(total, luminances[last])));
        for (std::size_t index = last; index-- > 0;) {
            const std::array<Value, fieldCount> candidate = fields(shader.bsdfs[index], o.mul(shader.bsdfs[index].weight, o.div(total, luminances[index])));
            const Value picked = o.less(random, bounds[index]);
            for (std::size_t field = 0; field < fieldCount; ++field)
                chosen[field] = o.select(picked, candidate[field], chosen[field]);
        }
    }
    for (std::size_t field = 0; field < 12; ++field)
        builder.setOutput(static_cast<GraphOutput>(field), chosen[field]);
    builder.setOutput(GraphOutput::Closure, o.combine({o.div(chosen[12], opacity), chosen[13]}));
    builder.setOutput(GraphOutput::Subsurface, chosen[14]);
    builder.setOutput(GraphOutput::Emission, shader.emission ? o.combine({o.div(*shader.emission, opacity), one}) : builder.constant(Core::float4{}));
    builder.setOutput(GraphOutput::Transparency, transparency);
}

std::optional<GraphCompiler::Result> GraphCompiler::compile(NodeRef output, const bool bumpsDisplacement) {
    if (bumpsDisplacement)
        bumpDisplacement(output);
    finish(shaderInput({}, output, "Surface"));
    return take();
}

std::optional<GraphCompiler::Result> GraphCompiler::compileLight(NodeRef output) {
    light = true;
    return compile(output);
}

std::optional<GraphCompiler::Result> GraphCompiler::compileValue(NodeRef node, const std::string_view identifier) {
    Ops o{builder};
    volume = true;
    finish({});
    const Value scalar = value({}, node, identifier);
    builder.setOutput(GraphOutput::Emission, o.combine({scalar, scalar, scalar, o.c(1.0f)}));
    return take();
}

std::optional<GraphCompiler::Result> GraphCompiler::compileDisplacement(NodeRef output, const Result& surface) {
    Ops o{builder};
    result.tangents = surface.tangents;
    result.vertexColors = surface.vertexColors;
    result.streams = surface.streams;
    result.tangentMap = surface.tangentMap;
    finish({});
    const Value offset = o.transform(rows({}, GraphInput::WorldToObject0), socketValue({}, output, input(output, "Displacement")), false);
    builder.setOutput(GraphOutput::Emission, o.combine({o.x(offset, 0), o.x(offset, 1), o.x(offset, 2), o.c(1.0f)}));
    return take();
}

std::optional<GraphCompiler::Result> GraphCompiler::take() {
    if (diagnostics.failed())
        return std::nullopt;
    result.parameters = builder.parameterValues();
    auto graph = std::make_shared<Materials::Graph>();
    if (std::expected<void, Core::Error> opened = graph->open(builder.take(), materialName); !opened) {
        diagnostics.fail(opened.error().message);
        return std::nullopt;
    }
    result.graph = std::move(graph);
    return std::move(result);
}

}
