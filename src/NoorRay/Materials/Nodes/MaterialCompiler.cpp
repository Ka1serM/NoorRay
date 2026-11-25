#include "NoorRay/Materials/Nodes/MaterialCompiler.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <memory>
#include <optional>
#include <string_view>

#include "NoorRay/Materials/Nodes/ProgramCompiler.h"
#include "NoorRay/Materials/Program.h"

namespace NoorRay::Materials {

namespace {

using Scene::TextureSlot;

struct PrincipledInput {
    std::string_view socket;
    float Scene::Material::* scalar = nullptr;
    Core::float3 Scene::Material::* color = nullptr;
    std::optional<TextureSlot> slot;
    //! An averaged emission would light a whole surface that the chain lights in places.
    bool folds = true;
};

constexpr PrincipledInput principledInputs[] = {
    {"Base Color", nullptr, &Scene::Material::baseColor, TextureSlot::BaseColor},
    {"Metallic", &Scene::Material::metallic, nullptr, TextureSlot::Metallic},
    {"Roughness", &Scene::Material::roughness, nullptr, TextureSlot::Roughness},
    {"IOR", &Scene::Material::ior, nullptr, std::nullopt},
    {"Alpha", &Scene::Material::alpha, nullptr, TextureSlot::Alpha},
    {"Specular IOR Level", &Scene::Material::specularIorLevel, nullptr, TextureSlot::Specular},
    {"Specular Tint", nullptr, &Scene::Material::specularTint, std::nullopt},
    {"Anisotropic", &Scene::Material::anisotropic, nullptr, std::nullopt},
    {"Anisotropic Rotation", &Scene::Material::anisotropicRotation, nullptr, std::nullopt},
    {"Transmission Weight", &Scene::Material::transmission, nullptr, TextureSlot::Transmission},
    {"Coat Weight", &Scene::Material::coat, nullptr, TextureSlot::Coat},
    {"Coat Roughness", &Scene::Material::coatRoughness, nullptr, TextureSlot::CoatRoughness},
    {"Sheen Roughness", &Scene::Material::sheenRoughness, nullptr, std::nullopt},
    {"Emission Color", nullptr, &Scene::Material::emissionColor, TextureSlot::Emission, false},
    {"Emission Strength", &Scene::Material::emissionStrength, nullptr, std::nullopt, false},
    {"Thin Film Thickness", &Scene::Material::thinFilmThickness, nullptr, std::nullopt},
    {"Thin Film IOR", &Scene::Material::thinFilmIor, nullptr, std::nullopt},
};

constexpr std::string_view ignoredPrincipledInputs[] = {
    "Weight", "Tangent", "Subsurface Radius", "Subsurface Scale", "Subsurface IOR", "Subsurface Anisotropy", "Coat IOR", "Coat Tint"};

const PrincipledInput* findPrincipledInput(const std::string_view name) {
    const auto found = std::ranges::find(principledInputs, name, &PrincipledInput::socket);
    return found == std::end(principledInputs) ? nullptr : found;
}

void assign(Scene::Material& material, const PrincipledInput& input, const Core::float3 value) {
    if (input.color)
        material.*input.color = value;
    else
        material.*input.scalar = value.x;
}

Core::float3 toFloat3(const NodeValues& value) {
    if (value.count == 1)
        return {value.values[0], value.values[0], value.values[0]};
    return {value.values[0], value.values[1], value.values[2]};
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

Scene::EmissionSampling toEmissionSampling(const std::string_view name) {
    if (name == "NONE")
        return Scene::EmissionSampling::Disabled;
    if (name == "FRONT")
        return Scene::EmissionSampling::Front;
    if (name == "BACK")
        return Scene::EmissionSampling::Back;
    if (name == "FRONT_AND_BACK")
        return Scene::EmissionSampling::FrontAndBack;
    return Scene::EmissionSampling::Auto;
}

Scene::TextureFilter toFilter(const std::string_view name) {
    if (name == "Closest")
        return Scene::TextureFilter::Closest;
    if (name == "Cubic" || name == "Smart")
        return Scene::TextureFilter::Cubic;
    return Scene::TextureFilter::Linear;
}

}

void MaterialCompiler::unsupported(const Record& record, const std::string& message) {
    record.diagnostics->unsupported(std::format("material '{}': {}", record.result.material.name, message));
}

CompiledMaterial MaterialCompiler::compile(const NodeTree& tree, const std::string& name) {
    Record record{.result = {}, .diagnostics = &diagnostics};
    Scene::Material& result = record.result.material;
    const std::span<const NodeProperty> settings = tree.properties;
    result.name = name;
    result.emissionSampling = toEmissionSampling(text(settings, "emission_sampling"));
    if (!flag(settings, "use_nodes")) {
        result.baseColor = toFloat3(values(settings, "diffuse_color"));
        result.metallic = number(settings, "metallic");
        result.roughness = number(settings, "roughness");
        return std::move(record.result);
    }
    const NodeRef output = findOutput(tree, "ShaderNodeOutputMaterial");
    if (!output) {
        result.baseColor = {};
        result.specularIorLevel = 0.0f;
        return std::move(record.result);
    }
    const NodeLinkEnd volumeLink = links.linkedNode(inputNamed(output, "Volume"));
    const NodeLinkEnd surfaceLink = links.linkedNode(inputNamed(output, "Surface"));
    if (!surfaceLink.node) {
        //! Cycles shades a material without any shader black.
        record.result.needs.hasSurface = !volumeLink.node;
        result.baseColor = {};
        result.specularIorLevel = 0.0f;
        return std::move(record.result);
    }
    GraphCompiler compiler(diagnostics, textures, attributeTables, result.name, static_cast<float>(integer(settings, "pass_index")));
    const std::string method = text(settings, "displacement_method");
    std::optional<GraphCompiler::Result> graph = compiler.compile(output, method == "BUMP" || method == "BOTH");
    if (!graph)
        return std::move(record.result);
    //! The fixed inputs stay for the light estimate and for renderers without graphs; the graph already reported what the material does not support.
    NodeDiagnostics quiet(true);
    record.diagnostics = &quiet;
    record.result.needs.streams = graph->streams;
    surface(record, surfaceLink.node);
    record.diagnostics = &diagnostics;
    result.graph = std::move(graph->graph);
    result.graphParameters = std::move(graph->parameters);
    result.graphTextures = std::move(graph->textures);
    displacement(record, tree, output, *graph);
    record.result.needs.streams = graph->streams;
    record.result.needs.tangentMap = graph->tangentMap;
    record.result.needs.tangents = graph->tangents;
    record.result.needs.vertexColors = graph->vertexColors;
    return std::move(record.result);
}

void MaterialCompiler::surface(Record& record, const NodeRef node) {
    const std::string_view type = nodeType(node);
    if (type == "BSDF_PRINCIPLED") {
        principled(record, node);
        return;
    }
    unsupported(record, std::format("the surface shader '{}' ({}) is not a Principled BSDF and is approximated", node->name, type));
    approximate(record, node);
}

void MaterialCompiler::principled(Record& record, const NodeRef node) {
    Scene::Material& material = record.result.material;
    const std::string& nodeName = node->name;
    for (const SocketRef socket : inputs(node)) {
        const std::string& name = socket->name;
        if (std::ranges::find(ignoredPrincipledInputs, name) != std::end(ignoredPrincipledInputs) || !socket->enabled)
            continue;
        const bool linked = links.isLinked(socket);
        if (const PrincipledInput* input = findPrincipledInput(name)) {
            assign(material, *input, toFloat3(socketValue(socket)));
            if (!linked)
                continue;
            const NodeLinkEnd source = links.linkedNode(socket);
            const std::string_view sourceType = source.node ? nodeType(source.node) : std::string_view();
            if (sourceType == "VERTEX_COLOR" && input->color == &Scene::Material::baseColor) {
                material.baseColor = {1.0f, 1.0f, 1.0f};
                material.useVertexColor = true;
                record.result.needs.vertexColors = true;
                continue;
            }
            if (input->slot && sourceType == "TEX_IMAGE" && input->color) {
                if (imageTexture(record, *input->slot, source.node, source.socket, true, 0))
                    material.*input->color = {1.0f, 1.0f, 1.0f};
                continue;
            }
            if (input->slot && scalarTexture(record, *input->slot, socket)) {
                assign(material, *input, {1.0f, 1.0f, 1.0f});
                continue;
            }
            const std::optional<Core::float3> averaged = input->folds ? folder.average(textures, socket, input->color ? 3 : 1) : std::nullopt;
            if (averaged) {
                assign(material, *input, *averaged);
                const std::string values =
                    input->color ? std::format("({:.3f}, {:.3f}, {:.3f})", averaged->x, averaged->y, averaged->z) : std::format("{:.3f}", averaged->x);
                record.diagnostics->warn(
                    std::format("material '{}': the input '{}' of '{}' is a node chain; it is its average, {}", material.name, name, nodeName, values));
                continue;
            }
            unsupported(record, std::format("the input '{}' of '{}' is linked to '{}'", name, nodeName, source.node ? source.node->name : std::string("?")));
            continue;
        }
        if (name == "Normal" || name == "Coat Normal") {
            normalMap(record, node, name == "Normal" ? "Normal" : "Coat Normal");
            continue;
        }
        if (name == "Sheen Weight") {
            const SocketRef tint = inputNamed(node, "Sheen Tint");
            if (linked || links.isLinked(tint))
                unsupported(record, std::format("the sheen of '{}' is linked", nodeName));
            material.sheenColor = toFloat3(socketValues(tint)) * socketNumber(socket);
            continue;
        }
        if (name == "Sheen Tint")
            continue;
        if (name == "Subsurface Weight") {
            if (linked || socketNumber(socket) > 0.0f)
                unsupported(record, std::format("'{}' uses subsurface scattering", nodeName));
            continue;
        }
        if (name == "Diffuse Roughness" && !linked) {
            if (socketNumber(socket) > 0.0f)
                unsupported(record, std::format("'{}' uses diffuse roughness", nodeName));
            continue;
        }
        if (linked)
            unsupported(record, std::format("the input '{}' of '{}' is linked", name, nodeName));
    }
    if (material.alpha < 1.0f || material.texture(TextureSlot::Alpha))
        material.alphaMode = Scene::AlphaMode::Blend;
}

void MaterialCompiler::approximate(Record& record, const NodeRef node) {
    Scene::Material& material = record.result.material;
    const std::string_view type = nodeType(node);
    if (type == "MIX_SHADER" || type == "ADD_SHADER") {
        for (const SocketRef socket : inputs(node)) {
            const NodeLinkEnd child = links.linkedNode(socket);
            if (child.node && socket->type == NodeSocketType::Shader) {
                surface(record, child.node);
                return;
            }
        }
        return;
    }
    const auto value = [&](const char* name) { return toFloat3(socketValue(inputNamed(node, name))); };
    if (inputNamed(node, "Color"))
        material.baseColor = value("Color");
    if (inputNamed(node, "Roughness"))
        material.roughness = value("Roughness").x;
    if (type == "BSDF_GLASS" || type == "BSDF_REFRACTION") {
        material.transmission = 1.0f;
        material.ior = value("IOR").x;
    } else if (type == "BSDF_GLOSSY" || type == "BSDF_ANISOTROPIC") {
        material.metallic = 1.0f;
        if (inputNamed(node, "Anisotropy")) {
            material.anisotropic = std::max(0.0f, value("Anisotropy").x);
            material.anisotropicRotation = value("Rotation").x;
        }
    } else if (type == "BSDF_DIFFUSE") {
        material.roughness = 1.0f;
        material.specularIorLevel = 0.0f;
    } else if (type == "EMISSION") {
        material.baseColor = {};
        material.emissionColor = value("Color");
        material.emissionStrength = value("Strength").x;
    }
}

void MaterialCompiler::normalMap(Record& record, const NodeRef node, const char* name) {
    const NodeLinkEnd source = links.linkedNode(inputNamed(node, name));
    if (!source.node)
        return;
    const bool normalMap = nodeType(source.node) == "NORMAL_MAP" && text(source.node, "space") == "TANGENT";
    const SocketRef strength = normalMap ? inputNamed(source.node, "Strength") : SocketRef();
    if (!normalMap || links.isLinked(strength)) {
        unsupported(record, std::format("the input '{}' of '{}' is not a constant tangent-space normal map", name, node->name));
        return;
    }
    const NodeLinkEnd image = links.linkedNode(inputNamed(source.node, "Color"));
    if (!image.node || nodeType(image.node) != "TEX_IMAGE") {
        unsupported(record, std::format("the normal map '{}' does not read an image", source.node->name));
        return;
    }
    const bool coat = std::string_view(name) == "Coat Normal";
    if (!imageTexture(record, coat ? TextureSlot::CoatNormal : TextureSlot::Normal, image.node, image.socket, false, 0))
        return;
    (coat ? record.result.material.coatNormalScale : record.result.material.normalScale) = socketNumber(strength);
    record.result.needs.tangents = true;
}

bool MaterialCompiler::imageTexture(Record& record, const TextureSlot slot, const NodeRef node, const SocketRef socket, const bool srgb,
                                    const std::uint32_t channel) {
    if (text(node, "image").empty()) {
        unsupported(record, std::format("the image node '{}' has no image", node->name));
        return false;
    }
    Scene::MaterialTexture texture;
    texture.channel = socket->name == "Alpha" ? 3 : channel;
    uvMapping(record, node, texture);
    const std::expected<NodeTexture, std::string> image = textures.load(*node.tree, node.index, srgb ? NodeTextureUse::Color : NodeTextureUse::Data);
    if (image)
        texture.texture = image->texture;
    const std::string extension = text(node, "extension");
    const std::optional<Scene::TextureWrap> wrap = toWrap(extension);
    if (!wrap) {
        unsupported(record, std::format("the image node '{}' has the extension {}", node->name, extension));
        return false;
    }
    texture.wrap = *wrap;
    texture.filter = toFilter(text(node, "interpolation"));
    if (!image || diagnostics.failed())
        return false;
    record.result.material.texture(slot) = texture;
    return true;
}

bool MaterialCompiler::scalarTexture(Record& record, const TextureSlot slot, const SocketRef socket) {
    NodeLinkEnd source = links.linkedNode(socket);
    std::uint32_t channel = 0;
    if (source.node && nodeType(source.node) == "SEPARATE_COLOR" && text(source.node, "mode") == "RGB") {
        for (const SocketRef output : outputs(source.node)) {
            if (output == source.socket)
                break;
            ++channel;
        }
        source = links.linkedNode(inputAt(source.node, 0));
    }
    if (!source.node || nodeType(source.node) != "TEX_IMAGE")
        return false;
    return imageTexture(record, slot, source.node, source.socket, false, channel);
}

void MaterialCompiler::uvMapping(Record& record, const NodeRef imageNode, Scene::MaterialTexture& texture) {
    NodeLinkEnd source = links.linkedNode(inputNamed(imageNode, "Vector"));
    if (source.node && nodeType(source.node) == "MAPPING") {
        const SocketRef location = inputNamed(source.node, "Location");
        const SocketRef rotation = inputNamed(source.node, "Rotation");
        const SocketRef scale = inputNamed(source.node, "Scale");
        if (text(source.node, "vector_type") != "POINT" || links.isLinked(location) || links.isLinked(rotation) || links.isLinked(scale)) {
            unsupported(record, std::format("the mapping node '{}' is not a constant point mapping", source.node->name));
        } else {
            const NodeValues offset = socketValues(location);
            const float angle = socketValues(rotation).values[2];
            const NodeValues size = socketValues(scale);
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            texture.transform.row0 = {size.values[0] * cosine, -size.values[1] * sine, offset.values[0]};
            texture.transform.row1 = {size.values[0] * sine, size.values[1] * cosine, offset.values[1]};
        }
        source = links.linkedNode(inputNamed(source.node, "Vector"));
    }
    if (source.node && nodeType(source.node) == "UVMAP") {
        const std::string map = text(source.node, "uv_map");
        const UvStreams& streams = record.result.needs.streams;
        const auto found = std::ranges::find(streams, std::optional(UvStream{map}));
        if (found != streams.end())
            texture.uvSet = static_cast<std::uint32_t>(found - streams.begin());
        return;
    }
    if (source.node && nodeType(source.node) != "TEX_COORD")
        unsupported(record, std::format("the texture coordinates of '{}' come from '{}'", imageNode->name, source.node->name));
}

void MaterialCompiler::displacement(Record& record, const NodeTree& tree, const NodeRef output, GraphCompiler::Result& graph) {
    if (!links.isLinked(inputNamed(output, "Displacement")))
        return;
    const std::string method = text(tree.properties, "displacement_method");
    record.result.material.bumpsSurface = method != "DISPLACEMENT";
    if (method == "BUMP")
        return;
    MaterialNeeds& needs = record.result.needs;
    Scene::Material& result = record.result.material;
    //! A program reads attributes in full precision and compiles no kernel, so the graph takes only what a program cannot express.
    ProgramCompiler compiler;
    std::expected<std::optional<Program::Source>, std::string> compiled = compiler.compile(output);
    if (diagnostics.failed() || (compiled && !*compiled))
        return;
    needs.displaces = true;
    result.bumpsRestNormal = method == "BOTH";
    if (compiled) {
        needs.attributes = (*compiled)->attributes;
        auto program = std::make_shared<Program>();
        if (std::expected<void, Core::Error> opened = program->open(std::move(**compiled), result.name); !opened) {
            diagnostics.fail(opened.error().message);
            return;
        }
        result.programRegisters.assign(program->getRegisters().begin(), program->getRegisters().end());
        result.program = std::move(program);
        return;
    }
    GraphCompiler graphCompiler(diagnostics, textures, attributeTables, result.name, static_cast<float>(integer(tree.properties, "pass_index")));
    std::optional<GraphCompiler::Result> displaced = graphCompiler.compileDisplacement(output, graph);
    if (!displaced)
        return;
    result.displacementGraph = std::move(displaced->graph);
    result.displacementParameters = std::move(displaced->parameters);
    result.displacementTextures = std::move(displaced->textures);
    graph.streams = displaced->streams;
    graph.tangentMap = displaced->tangentMap;
    graph.tangents = displaced->tangents;
    graph.vertexColors = displaced->vertexColors;
}

}
