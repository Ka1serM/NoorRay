#include "NoorRay/Materials/Nodes/LightCompiler.h"

#include <cmath>
#include <numbers>
#include <optional>
#include <string>

#include "NoorRay/Materials/Nodes/GraphCompiler.h"
#include "NoorRay/Materials/Nodes/NodeAccess.h"

namespace NoorRay::Materials {

namespace {

Core::float3 toFloat3(const NodeValues& value) {
    return {value.values[0], value.values[1], value.values[2]};
}

//! The turn about z of the direction that a world texture reads, or none when its vector is more than a turn of the direction.
std::optional<float> worldTurn(NodeLinks& links, const SocketRef vector) {
    const NodeLinkEnd link = links.linkedNode(vector);
    if (!link.node)
        return 0.0f;
    const std::string_view type = nodeType(link.node);
    if (type == "TEX_COORD" && link.socket->name == "Generated")
        return 0.0f;
    if (type != "MAPPING" || link.node->muted || text(link.node, "vector_type") != "POINT")
        return std::nullopt;
    for (const char* name : {"Location", "Rotation", "Scale"})
        if (links.isLinked(inputNamed(link.node, name)))
            return std::nullopt;
    const Core::float3 location = toFloat3(socketValues(inputNamed(link.node, "Location")));
    const Core::float3 rotation = toFloat3(socketValues(inputNamed(link.node, "Rotation")));
    const Core::float3 scale = toFloat3(socketValues(inputNamed(link.node, "Scale")));
    const std::optional<float> turn = worldTurn(links, inputNamed(link.node, "Vector"));
    if (!turn || location != Core::float3{} || scale != Core::float3{1.0f, 1.0f, 1.0f} || rotation.x != 0.0f || rotation.y != 0.0f)
        return std::nullopt;
    return *turn + rotation.z;
}

//! Reads a background of a constant, an image or a sky; false for any other tree, which the device bakes.
bool readBackground(NodeLinks& links, NodeTextureSource& textures, const NodeRef node, Scene::Environment& environment) {
    if (nodeType(node) != "BACKGROUND" || node->muted)
        return false;
    const SocketRef color = inputNamed(node, "Color");
    const SocketRef strength = inputNamed(node, "Strength");
    if (links.isLinked(strength))
        return false;
    environment.strength = socketNumber(strength);
    environment.color = toFloat3(socketValue(color));
    const NodeLinkEnd source = links.linkedNode(color);
    if (!source.node)
        return true;
    const std::string_view type = nodeType(source.node);
    const SocketRef vector = inputNamed(source.node, "Vector");
    const std::optional<float> turn = vector ? worldTurn(links, vector) : std::optional(0.0f);
    if (!turn || source.node->muted)
        return false;
    environment.rotation -= *turn;
    environment.color = {1.0f, 1.0f, 1.0f};
    if (type == "TEX_SKY") {
        const std::expected<NodeTexture, std::string> sky = textures.load(*source.node.tree, source.node.index, NodeTextureUse::WorldSky);
        environment.texture = sky ? sky->texture : Scene::TextureHandle();
        return true;
    }
    if (type != "TEX_ENVIRONMENT" || text(source.node, "image").empty() || text(source.node, "projection") != "EQUIRECTANGULAR"
        || source.socket->name != "Color")
        return false;
    const std::expected<NodeTexture, std::string> image = textures.load(*source.node.tree, source.node.index, NodeTextureUse::Background);
    environment.texture = image ? image->texture : Scene::TextureHandle();
    environment.filter = text(source.node, "interpolation") == "Closest" ? Scene::TextureFilter::Closest : Scene::TextureFilter::Linear;
    return true;
}

}

void compileLight(const NodeTree& tree, Scene::Light& light, NodeDiagnostics& diagnostics, NodeTextureSource& textures) {
    const std::span<const NodeProperty> settings = tree.properties;
    Core::float3 shaderColor{1.0f, 1.0f, 1.0f};
    float shaderStrength = 1.0f;
    light.falloffPower = 0;
    light.falloffSmooth = 0.0f;
    light.graph = nullptr;
    light.graphParameters.clear();
    light.graphTextures.clear();
    if (flag(settings, "use_nodes")) {
        NodeLinks links;
        const NodeRef output = findOutput(tree, "ShaderNodeOutputLight");
        const NodeLinkEnd source = output ? links.linkedNode(inputNamed(output, "Surface")) : NodeLinkEnd{};
        const bool emission = source.node && nodeType(source.node) == "EMISSION";
        const SocketRef color = emission ? inputNamed(source.node, "Color") : SocketRef();
        const SocketRef strength = emission ? inputNamed(source.node, "Strength") : SocketRef();
        if (emission && !links.isLinked(color) && !links.isLinked(strength)) {
            shaderColor = toFloat3(socketValues(color));
            shaderStrength = socketNumber(strength);
        } else if (source.node) {
            //! The device bakes the light shader over the light UV, and its emission multiplies the light like the shader of a Cycles light.
            GraphCompiler compiler(diagnostics, textures, nullptr, light.name);
            if (std::optional<GraphCompiler::Result> compiled = compiler.compileLight(output)) {
                light.falloffPower = compiled->falloffPower;
                light.falloffSmooth = compiled->falloffSmooth;
                light.graph = std::move(compiled->graph);
                light.graphParameters = std::move(compiled->parameters);
                light.graphTextures = std::move(compiled->textures);
            }
        }
    }
    light.power = number(settings, "energy") * shaderStrength * std::exp2(number(settings, "exposure"));
    light.color = toFloat3(values(settings, "color")) * shaderColor;
    if (flag(settings, "use_temperature"))
        light.color = light.color * toFloat3(values(settings, "temperature_color"));
    const bool round = light.type == Scene::LightType::Disk;
    const bool area = round || light.type == Scene::LightType::Rect;
    if (area && findProperty(settings, "normalize") && !flag(settings, "normalize"))
        light.power *= light.width * light.height * (round ? Core::pi / 4.0f : 1.0f);
}

Scene::Environment compileEnvironment(const NodeTree& tree, const Scene::Environment& last, NodeDiagnostics& diagnostics, NodeTextureSource& textures) {
    const std::span<const NodeProperty> settings = tree.properties;
    Scene::Environment environment;
    environment.opaque = !flag(settings, "film_transparent");
    environment.visibleToCamera = environment.opaque;
    //! Blender's equirect u is 0.5 - atan2(y, x) / 2 pi; the NoorRay map turned by -pi/2 about +Y has the same u.
    environment.rotation = -std::numbers::pi_v<float> / 2.0f + number(settings, "noorray.rotation");
    NodeLinks links;
    const NodeRef output = flag(settings, "use_nodes") ? findOutput(tree, "ShaderNodeOutputWorld") : NodeRef();
    const NodeLinkEnd surface = output ? links.linkedNode(inputNamed(output, "Surface")) : NodeLinkEnd{};
    if (surface.node && !readBackground(links, textures, surface.node, environment)) {
        environment.texture = {};
        environment.color = {1.0f, 1.0f, 1.0f};
        environment.strength = 1.0f;
        environment.rotation = 0.0f;
        environment.filter = Scene::TextureFilter::Linear;
        GraphCompiler compiler(diagnostics, textures, nullptr, text(settings, "name"));
        if (std::optional<GraphCompiler::Result> compiled = compiler.compile(output)) {
            const std::optional<Core::float4> uniform =
                compiled->graph->evaluateUniform(compiled->graph->getOutput(GraphOutput::Emission), compiled->parameters);
            const bool same = last.graph && last.graph->hasSameProgram(*compiled->graph) && last.graphParameters == compiled->parameters
                              && last.graphTextures == compiled->textures;
            if (uniform) {
                environment.color = {uniform->x, uniform->y, uniform->z};
            } else {
                environment.graph = same ? last.graph : std::move(compiled->graph);
                environment.graphParameters = std::move(compiled->parameters);
                environment.graphTextures = std::move(compiled->textures);
            }
        }
    }
    if (findProperty(settings, "cycles_visibility.camera")) {
        environment.visibleToCamera = environment.visibleToCamera && flag(settings, "cycles_visibility.camera");
        constexpr const char* rays[] = {"cycles_visibility.diffuse", "cycles_visibility.glossy", "cycles_visibility.transmission", "cycles_visibility.scatter"};
        for (std::uint32_t ray = 0; ray < std::size(rays); ++ray)
            environment.hidden |= flag(settings, rays[ray]) ? 0u : 1u << ray;
    }
    //! Cycles' Auto samples a world only where it varies, so a constant colour is found by scattered rays alone, which changes what the clamps cut.
    const std::string sampling = findProperty(settings, "sampling_method") ? text(settings, "sampling_method") : std::string("AUTOMATIC");
    environment.sampled = sampling == "MANUAL" || (sampling == "AUTOMATIC" && (environment.texture || environment.graph));
    return environment;
}

}
