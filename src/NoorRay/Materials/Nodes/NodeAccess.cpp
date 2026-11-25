#include "NoorRay/Materials/Nodes/NodeAccess.h"

#include <algorithm>
#include <utility>

namespace NoorRay::Materials {

namespace {

constexpr std::pair<std::string_view, std::string_view> nodeTypes[] = {
    {"NodeClosureInput", "NodeClosureInput"},
    {"NodeClosureOutput", "NodeClosureOutput"},
    {"NodeCombineBundle", "NodeCombineBundle"},
    {"NodeEvaluateClosure", "NodeEvaluateClosure"},
    {"NodeFrame", "FRAME"},
    {"NodeGroupInput", "GROUP_INPUT"},
    {"NodeGroupOutput", "GROUP_OUTPUT"},
    {"NodeImplicitConversion", "NodeImplicitConversion"},
    {"NodeJoinBundle", "NodeJoinBundle"},
    {"NodeReroute", "REROUTE"},
    {"NodeSeparateBundle", "NodeSeparateBundle"},
    {"ShaderNodeAddShader", "ADD_SHADER"},
    {"ShaderNodeAmbientOcclusion", "AMBIENT_OCCLUSION"},
    {"ShaderNodeAttribute", "ATTRIBUTE"},
    {"ShaderNodeBackground", "BACKGROUND"},
    {"ShaderNodeBevel", "BEVEL"},
    {"ShaderNodeBlackbody", "BLACKBODY"},
    {"ShaderNodeBrightContrast", "BRIGHTCONTRAST"},
    {"ShaderNodeBsdfAnisotropic", "BSDF_GLOSSY"},
    {"ShaderNodeBsdfDiffuse", "BSDF_DIFFUSE"},
    {"ShaderNodeBsdfGlass", "BSDF_GLASS"},
    {"ShaderNodeBsdfHair", "BSDF_HAIR"},
    {"ShaderNodeBsdfHairPrincipled", "BSDF_HAIR_PRINCIPLED"},
    {"ShaderNodeBsdfMetallic", "BSDF_METALLIC"},
    {"ShaderNodeBsdfPrincipled", "BSDF_PRINCIPLED"},
    {"ShaderNodeBsdfRayPortal", "BSDF_RAY_PORTAL"},
    {"ShaderNodeBsdfRefraction", "BSDF_REFRACTION"},
    {"ShaderNodeBsdfSheen", "BSDF_SHEEN"},
    {"ShaderNodeBsdfToon", "BSDF_TOON"},
    {"ShaderNodeBsdfTranslucent", "BSDF_TRANSLUCENT"},
    {"ShaderNodeBsdfTransparent", "BSDF_TRANSPARENT"},
    {"ShaderNodeBump", "BUMP"},
    {"ShaderNodeCameraData", "CAMERA"},
    {"ShaderNodeClamp", "CLAMP"},
    {"ShaderNodeCombineColor", "COMBINE_COLOR"},
    {"ShaderNodeCombineXYZ", "COMBXYZ"},
    {"ShaderNodeDisplacement", "DISPLACEMENT"},
    {"ShaderNodeEeveeSpecular", "EEVEE_SPECULAR"},
    {"ShaderNodeEmission", "EMISSION"},
    {"ShaderNodeFloatCurve", "CURVE_FLOAT"},
    {"ShaderNodeFresnel", "FRESNEL"},
    {"ShaderNodeGamma", "GAMMA"},
    {"ShaderNodeGroup", "GROUP"},
    {"ShaderNodeHairInfo", "HAIR_INFO"},
    {"ShaderNodeHoldout", "HOLDOUT"},
    {"ShaderNodeHueSaturation", "HUE_SAT"},
    {"ShaderNodeInvert", "INVERT"},
    {"ShaderNodeLayerWeight", "LAYER_WEIGHT"},
    {"ShaderNodeLightFalloff", "LIGHT_FALLOFF"},
    {"ShaderNodeLightPath", "LIGHT_PATH"},
    {"ShaderNodeMapRange", "MAP_RANGE"},
    {"ShaderNodeMapping", "MAPPING"},
    {"ShaderNodeMath", "MATH"},
    {"ShaderNodeMix", "MIX"},
    {"ShaderNodeMixRGB", "MIX_RGB"},
    {"ShaderNodeMixShader", "MIX_SHADER"},
    {"ShaderNodeNewGeometry", "NEW_GEOMETRY"},
    {"ShaderNodeNormal", "NORMAL"},
    {"ShaderNodeNormalMap", "NORMAL_MAP"},
    {"ShaderNodeObjectInfo", "OBJECT_INFO"},
    {"ShaderNodeOutputAOV", "OUTPUT_AOV"},
    {"ShaderNodeOutputLight", "OUTPUT_LIGHT"},
    {"ShaderNodeOutputLineStyle", "OUTPUT_LINESTYLE"},
    {"ShaderNodeOutputMaterial", "OUTPUT_MATERIAL"},
    {"ShaderNodeOutputWorld", "OUTPUT_WORLD"},
    {"ShaderNodeParticleInfo", "PARTICLE_INFO"},
    {"ShaderNodePointInfo", "POINT_INFO"},
    {"ShaderNodeRGB", "RGB"},
    {"ShaderNodeRGBCurve", "CURVE_RGB"},
    {"ShaderNodeRGBToBW", "RGBTOBW"},
    {"ShaderNodeRadialTiling", "ShaderNodeRadialTiling"},
    {"ShaderNodeRaycast", "MATERIAL_RAYCAST"},
    {"ShaderNodeScript", "SCRIPT"},
    {"ShaderNodeSeparateColor", "SEPARATE_COLOR"},
    {"ShaderNodeSeparateXYZ", "SEPXYZ"},
    {"ShaderNodeShaderToRGB", "SHADERTORGB"},
    {"ShaderNodeSqueeze", "SQUEEZE"},
    {"ShaderNodeSubsurfaceScattering", "SUBSURFACE_SCATTERING"},
    {"ShaderNodeTangent", "TANGENT"},
    {"ShaderNodeTexBrick", "TEX_BRICK"},
    {"ShaderNodeTexChecker", "TEX_CHECKER"},
    {"ShaderNodeTexCoord", "TEX_COORD"},
    {"ShaderNodeTexEnvironment", "TEX_ENVIRONMENT"},
    {"ShaderNodeTexGabor", "TEX_GABOR"},
    {"ShaderNodeTexGradient", "TEX_GRADIENT"},
    {"ShaderNodeTexIES", "TEX_IES"},
    {"ShaderNodeTexImage", "TEX_IMAGE"},
    {"ShaderNodeTexMagic", "TEX_MAGIC"},
    {"ShaderNodeTexNoise", "TEX_NOISE"},
    {"ShaderNodeTexSky", "TEX_SKY"},
    {"ShaderNodeTexVoronoi", "TEX_VORONOI"},
    {"ShaderNodeTexWave", "TEX_WAVE"},
    {"ShaderNodeTexWhiteNoise", "TEX_WHITE_NOISE"},
    {"ShaderNodeUVAlongStroke", "UVALONGSTROKE"},
    {"ShaderNodeUVMap", "UVMAP"},
    {"ShaderNodeValToRGB", "VALTORGB"},
    {"ShaderNodeValue", "VALUE"},
    {"ShaderNodeVectorCurve", "CURVE_VEC"},
    {"ShaderNodeVectorDisplacement", "VECTOR_DISPLACEMENT"},
    {"ShaderNodeVectorMath", "VECT_MATH"},
    {"ShaderNodeVectorRotate", "VECTOR_ROTATE"},
    {"ShaderNodeVectorTransform", "VECT_TRANSFORM"},
    {"ShaderNodeVertexColor", "VERTEX_COLOR"},
    {"ShaderNodeVolumeAbsorption", "VOLUME_ABSORPTION"},
    {"ShaderNodeVolumeCoefficients", "VOLUME_COEFFICIENTS"},
    {"ShaderNodeVolumeInfo", "VOLUME_INFO"},
    {"ShaderNodeVolumePrincipled", "PRINCIPLED_VOLUME"},
    {"ShaderNodeVolumeScatter", "VOLUME_SCATTER"},
    {"ShaderNodeWavelength", "WAVELENGTH"},
    {"ShaderNodeWireframe", "WIREFRAME"},
};

std::uint64_t socketKey(const std::uint32_t node, const std::uint32_t socket) {
    return std::uint64_t{node} << 32 | socket;
}

SocketRef find(const NodeRef node, const bool output, const std::string_view value, const bool byName) {
    const std::vector<NodeSocket>& sockets = output ? node->outputs : node->inputs;
    for (std::size_t index = 0; index < sockets.size(); ++index)
        if (byName ? sockets[index].name == value && sockets[index].enabled : sockets[index].identifier == value)
            return {node.tree, node.index, static_cast<std::uint32_t>(index), output};
    return {};
}

std::vector<SocketRef> all(const NodeRef node, const bool output) {
    std::vector<SocketRef> result;
    const std::size_t count = output ? node->outputs.size() : node->inputs.size();
    for (std::size_t index = 0; index < count; ++index)
        result.push_back({node.tree, node.index, static_cast<std::uint32_t>(index), output});
    return result;
}

}

std::string_view nodeType(const std::string_view idname) {
    const auto found = std::ranges::lower_bound(nodeTypes, idname, {}, &std::pair<std::string_view, std::string_view>::first);
    if (found == std::end(nodeTypes) || found->first != idname)
        return "CUSTOM";
    return found->second;
}

std::string_view socketTypeName(const NodeSocketType type) {
    constexpr std::string_view names[] = {"VALUE", "INT", "BOOLEAN", "VECTOR", "RGBA", "SHADER", "STRING", "ROTATION", "MATRIX", "MENU", "CLOSURE", "CUSTOM"};
    return names[static_cast<std::size_t>(type)];
}

SocketRef input(const NodeRef node, const std::string_view identifier) {
    return find(node, false, identifier, false);
}

SocketRef output(const NodeRef node, const std::string_view identifier) {
    return find(node, true, identifier, false);
}

SocketRef inputAt(const NodeRef node, const std::size_t index) {
    if (index >= node->inputs.size())
        return {};
    return {node.tree, node.index, static_cast<std::uint32_t>(index), false};
}

SocketRef outputAt(const NodeRef node, const std::size_t index) {
    if (index >= node->outputs.size())
        return {};
    return {node.tree, node.index, static_cast<std::uint32_t>(index), true};
}

SocketRef inputNamed(const NodeRef node, const std::string_view name) {
    return find(node, false, name, true);
}

SocketRef outputNamed(const NodeRef node, const std::string_view name) {
    return find(node, true, name, true);
}

std::vector<SocketRef> inputs(const NodeRef node) {
    return all(node, false);
}

std::vector<SocketRef> outputs(const NodeRef node) {
    return all(node, true);
}

NodeValues socketValues(const SocketRef socket) {
    if (!socket)
        return {};
    const NodeSocket& value = socket.get();
    return {{value.value.x, value.value.y, value.value.z, value.value.w}, value.components};
}

float socketNumber(const SocketRef socket) {
    return socketValues(socket).values[0];
}

NodeValues socketValue(const SocketRef socket) {
    NodeValues value = socketValues(socket);
    value.count = std::min<std::size_t>(value.count, 3);
    return value;
}

const NodeProperty* findProperty(const std::span<const NodeProperty> properties, const std::string_view name) {
    const auto found = std::ranges::find(properties, name, &NodeProperty::name);
    return found == properties.end() ? nullptr : &*found;
}

std::string text(const std::span<const NodeProperty> properties, const std::string_view name) {
    const NodeProperty* property = findProperty(properties, name);
    return property ? property->text : std::string();
}

float number(const std::span<const NodeProperty> properties, const std::string_view name, const float fallback) {
    const NodeProperty* property = findProperty(properties, name);
    if (!property)
        return fallback;
    if (property->kind == NodePropertyKind::Integer || property->kind == NodePropertyKind::Boolean)
        return static_cast<float>(property->integer);
    return property->number.x;
}

NodeValues values(const std::span<const NodeProperty> properties, const std::string_view name) {
    const NodeProperty* property = findProperty(properties, name);
    if (!property)
        return {};
    if (property->kind == NodePropertyKind::Numbers) {
        NodeValues result;
        result.count = std::min<std::size_t>(property->numbers.size(), 4);
        std::copy_n(property->numbers.begin(), result.count, result.values.begin());
        return result;
    }
    return {{property->number.x, property->number.y, property->number.z, property->number.w}, 1};
}

std::int64_t integer(const std::span<const NodeProperty> properties, const std::string_view name) {
    const NodeProperty* property = findProperty(properties, name);
    if (!property)
        return 0;
    if (property->kind == NodePropertyKind::Number)
        return static_cast<std::int64_t>(property->number.x);
    return property->integer;
}

bool flag(const std::span<const NodeProperty> properties, const std::string_view name) {
    return integer(properties, name) != 0;
}

std::span<const float> numbers(const std::span<const NodeProperty> properties, const std::string_view name) {
    const NodeProperty* property = findProperty(properties, name);
    if (!property)
        return {};
    return property->numbers;
}

const NodeLinks::Table& NodeLinks::table(const NodeTree& tree) {
    const auto [found, added] = tables.try_emplace(&tree);
    if (!added)
        return found->second;
    for (std::size_t index = 0; index < tree.links.size(); ++index) {
        const NodeLink& link = tree.links[index];
        found->second.inputs.try_emplace(socketKey(link.toNode, link.toSocket), static_cast<std::uint32_t>(index));
        found->second.outputs.try_emplace(socketKey(link.fromNode, link.fromSocket), static_cast<std::uint32_t>(index));
    }
    return found->second;
}

bool NodeLinks::isLinked(const SocketRef socket) {
    if (!socket)
        return false;
    const Table& links = table(*socket.tree);
    return (socket.output ? links.outputs : links.inputs).contains(socketKey(socket.node, socket.index));
}

NodeLinkEnd NodeLinks::linkedNode(const SocketRef input) {
    if (!input)
        return {};
    const Table* links = &table(*input.tree);
    auto found = links->inputs.find(socketKey(input.node, input.index));
    if (found == links->inputs.end())
        return {};
    const NodeTree& tree = *input.tree;
    const NodeLink* link = &tree.links[found->second];
    if (link->muted)
        return {};
    while (nodeType(tree.nodes[link->fromNode].type) == "REROUTE") {
        found = links->inputs.find(socketKey(link->fromNode, 0));
        if (found == links->inputs.end())
            return {};
        link = &tree.links[found->second];
    }
    return {{&tree, link->fromNode}, {&tree, link->fromNode, link->fromSocket, true}};
}

RampPieces rampPieces(const std::span<const float> elements) {
    constexpr std::size_t pieces = 8;
    std::vector<float> knots{0.0f, 1.0f};
    for (std::size_t index = 0; index + rampElementSize <= elements.size(); index += rampElementSize)
        knots.push_back(elements[index]);
    std::ranges::sort(knots);
    knots.erase(std::ranges::unique(knots).begin(), knots.end());
    RampPieces result{knots.front(), {}};
    for (std::size_t knot = 0; knot + 1 < knots.size(); ++knot)
        for (std::size_t piece = 0; piece < pieces; ++piece) {
            const float step = (knots[knot + 1] - knots[knot]) / static_cast<float>(pieces);
            result.pieces.push_back({knots[knot] + step * static_cast<float>(piece), step});
        }
    return result;
}

std::pair<float, float> curveRange(const std::initializer_list<std::span<const float>> curves) {
    float low = 1e30f;
    float high = -1e30f;
    for (const std::span<const float> points : curves)
        for (std::size_t index = 0; index + 1 < points.size(); index += 2) {
            low = std::min(low, points[index]);
            high = std::max(high, points[index]);
        }
    return low <= high ? std::pair{low, high} : std::pair{0.0f, 1.0f};
}

std::vector<float> curveSamplePositions(const float low, const float high) {
    if (high - low < 1e-6f)
        return {low};
    const float step = (high - low) / static_cast<float>(curveSamples - 1);
    std::vector<float> positions{low};
    for (std::size_t segment = 0; segment + 1 < curveSamples; ++segment)
        positions.push_back(low + step * static_cast<float>(segment + 1));
    return positions;
}

const NodeTree* groupTree(const NodeRef node) {
    if (node->group == noNode || node->group >= node.tree->groups.size())
        return nullptr;
    return node.tree->groups[node->group].get();
}

bool isSameTree(const NodeTree& a, const NodeTree& b) {
    if (&a == &b)
        return true;
    if (a.name != b.name || a.nodes != b.nodes || a.links != b.links || a.properties != b.properties || a.groups.size() != b.groups.size())
        return false;
    for (std::size_t index = 0; index < a.groups.size(); ++index) {
        if (!a.groups[index] || !b.groups[index]) {
            if (a.groups[index] != b.groups[index])
                return false;
            continue;
        }
        if (!isSameTree(*a.groups[index], *b.groups[index]))
            return false;
    }
    return true;
}

NodeRef findOutput(const NodeTree& tree, const std::string_view idname) {
    NodeRef first;
    for (std::uint32_t index = 0; index < tree.nodes.size(); ++index) {
        const Node& node = tree.nodes[index];
        if (node.type != idname)
            continue;
        const std::string target = text(node.properties, "target");
        if (target != "ALL" && target != "CYCLES")
            continue;
        if (flag(node.properties, "is_active_output"))
            return {&tree, index};
        if (!first)
            first = {&tree, index};
    }
    return first;
}

NodeRef findGroupOutput(const NodeTree& tree) {
    NodeRef result;
    for (std::uint32_t index = 0; index < tree.nodes.size(); ++index)
        if (tree.nodes[index].type == "NodeGroupOutput" && (!result || flag(tree.nodes[index].properties, "is_active_output")))
            result = {&tree, index};
    return result;
}

}
