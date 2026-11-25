#include "NoorRay/Import/PbrtImporter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include "NoorRay/Core/Log.h"
#include "NoorRay/Core/MappedFile.h"
#include "NoorRay/Import/MeshBuilder.h"
#include "NoorRay/Import/PbrtParser.h"
#include "NoorRay/Import/PlyFile.h"
#include "NoorRay/Materials/TextureCache.h"
#include "NoorRay/Scene/Scene.h"
#include "NoorRay/Spectral/Blackbody.h"

namespace NoorRay::Import {

namespace {

using Pbrt::Command;
using Pbrt::Parameter;

constexpr float lumensPerWatt = 683.0f;

Core::Transform fromColumns(const Core::float3 x, const Core::float3 y, const Core::float3 z, const Core::float3 t) {
    Core::Transform transform;
    const Core::float3 columns[] = {x, y, z, t};
    for (int column = 0; column < 4; ++column) {
        transform.rows[0][column] = columns[column].x;
        transform.rows[1][column] = columns[column].y;
        transform.rows[2][column] = columns[column].z;
    }
    return transform;
}

Core::Transform rotation(const float degrees, const Core::float3 axis) {
    const Core::float3 a = Core::normalize(axis);
    const float s = std::sin(degrees * Core::pi / 180.0f);
    const float c = std::cos(degrees * Core::pi / 180.0f);
    Core::Transform transform;
    transform.rows[0][0] = a.x * a.x + (1.0f - a.x * a.x) * c;
    transform.rows[0][1] = a.x * a.y * (1.0f - c) - a.z * s;
    transform.rows[0][2] = a.x * a.z * (1.0f - c) + a.y * s;
    transform.rows[1][0] = a.x * a.y * (1.0f - c) + a.z * s;
    transform.rows[1][1] = a.y * a.y + (1.0f - a.y * a.y) * c;
    transform.rows[1][2] = a.y * a.z * (1.0f - c) - a.x * s;
    transform.rows[2][0] = a.x * a.z * (1.0f - c) - a.y * s;
    transform.rows[2][1] = a.y * a.z * (1.0f - c) + a.x * s;
    transform.rows[2][2] = a.z * a.z + (1.0f - a.z * a.z) * c;
    return transform;
}

//! The frame whose -Z axis points along a direction, placed at a position.
Core::Transform facing(const Core::float3 position, const Core::float3 direction) {
    const Core::float3 back = Core::normalize(-direction);
    const Core::float3 helper = std::abs(back.y) < 0.99f ? Core::float3{0.0f, 1.0f, 0.0f} : Core::float3{1.0f, 0.0f, 0.0f};
    const Core::float3 right = Core::normalize(Core::cross(helper, back));
    return fromColumns(right, Core::cross(back, right), back, position);
}

std::optional<std::vector<float>> parseNumbers(const std::vector<std::string>& values) {
    std::vector<float> result;
    for (const std::string& value : values) {
        Parameter parameter{"float", "", {value}};
        const std::optional<float> number = parameter.getFloat();
        if (!number)
            return std::nullopt;
        result.push_back(*number);
    }
    return result;
}

Core::float3 getVector(const Command& command, const std::string_view name, const Core::float3 fallback) {
    const Parameter* parameter = command.find(name);
    if (!parameter)
        return fallback;
    const std::vector<float> values = parameter->getFloats();
    if (values.size() >= 3)
        return {values[0], values[1], values[2]};
    if (values.size() == 1)
        return {values[0], values[0], values[0]};
    return fallback;
}

//! RGB, blackbody and the named spectra pbrt scenes use most, reduced to linear sRGB.
std::optional<Core::float3> getColor(const Command& command, const std::string_view name) {
    const Parameter* parameter = command.find(name);
    if (!parameter || parameter->values.empty())
        return std::nullopt;
    if (parameter->type == "blackbody") {
        const std::optional<float> kelvin = parameter->getFloat();
        return kelvin ? std::optional(Spectral::blackbodyColor(*kelvin)) : std::nullopt;
    }
    if (parameter->type == "spectrum") {
        if (const std::optional<std::vector<float>> values = parseNumbers(parameter->values); values && values->size() >= 2) {
            float sum = 0.0f;
            for (std::size_t index = 1; index < values->size(); index += 2)
                sum += (*values)[index];
            const float mean = sum / static_cast<float>(values->size() / 2);
            return Core::float3{mean, mean, mean};
        }
        return Core::float3{1.0f, 1.0f, 1.0f};
    }
    const std::vector<float> values = parameter->getFloats();
    if (values.size() >= 3)
        return Core::float3{values[0], values[1], values[2]};
    if (values.size() == 1)
        return Core::float3{values[0], values[0], values[0]};
    return std::nullopt;
}

//! Normal-incidence reflectance of the conductor spectra that pbrt names.
std::optional<Core::float3> getNamedConductor(const std::string& name) {
    static const std::unordered_map<std::string_view, Core::float3> conductors{
        {"metal-Au", {1.0f, 0.71f, 0.29f}},    {"metal-Ag", {0.95f, 0.93f, 0.88f}}, {"metal-Cu", {0.95f, 0.64f, 0.54f}},   {"metal-Al", {0.91f, 0.92f, 0.92f}},
        {"metal-CuZn", {0.94f, 0.78f, 0.51f}}, {"metal-MgO", {0.2f, 0.2f, 0.2f}},   {"metal-TiO2", {0.25f, 0.25f, 0.25f}},
    };
    const std::string_view base = std::string_view(name).substr(0, name.rfind('-'));
    const auto found = conductors.find(base);
    return found == conductors.end() ? std::nullopt : std::optional(found->second);
}

struct NamedGlass {
    float ior = 1.5f;
    float abbe = 0.0f;
};

std::optional<NamedGlass> getNamedGlass(const std::string& name) {
    static const std::unordered_map<std::string_view, NamedGlass> glasses{
        {"glass-BK7", {1.5168f, 64.17f}}, {"glass-BAF10", {1.6700f, 47.11f}}, {"glass-FK51A", {1.4866f, 84.47f}}, {"glass-LASF9", {1.8503f, 32.17f}},
        {"glass-F5", {1.6034f, 38.03f}},  {"glass-F10", {1.7283f, 28.41f}},   {"glass-F11", {1.7847f, 25.68f}},
    };
    const auto found = glasses.find(name);
    return found == glasses.end() ? std::nullopt : std::optional(found->second);
}

Core::float3 fresnelConductor(const Core::float3 eta, const Core::float3 k) {
    const auto channel = [](const float n, const float kk) { return ((n - 1.0f) * (n - 1.0f) + kk * kk) / ((n + 1.0f) * (n + 1.0f) + kk * kk); };
    return {channel(eta.x, k.x), channel(eta.y, k.y), channel(eta.z, k.z)};
}

struct TextureDefinition {
    std::filesystem::path path;
    bool linear = false;
    Scene::UvTransform transform;
};

struct State {
    Core::Transform transform;
    std::int32_t material = 0;
    std::optional<Core::float3> emission;
    bool reverse = false;
};

struct ShapeRecord {
    const Command* command = nullptr;
    Core::Transform transform;
    std::int32_t material = 0;
    std::optional<Core::float3> emission;
    bool reverse = false;
};

struct CameraRecord {
    const Command* command = nullptr;
    Core::Transform cameraFromWorld;
};

std::vector<Core::float3> makeSphere(const float radius, std::vector<std::uint32_t>& indices, std::vector<Core::float2>& uvs) {
    constexpr std::uint32_t rings = 48;
    constexpr std::uint32_t segments = 96;
    std::vector<Core::float3> positions;
    for (std::uint32_t ring = 0; ring <= rings; ++ring) {
        const float theta = Core::pi * static_cast<float>(ring) / rings;
        for (std::uint32_t segment = 0; segment <= segments; ++segment) {
            const float phi = 2.0f * Core::pi * static_cast<float>(segment) / segments;
            positions.push_back({radius * std::sin(theta) * std::cos(phi), radius * std::sin(theta) * std::sin(phi), radius * std::cos(theta)});
            uvs.push_back({static_cast<float>(segment) / segments, 1.0f - static_cast<float>(ring) / rings});
        }
    }
    for (std::uint32_t ring = 0; ring < rings; ++ring) {
        for (std::uint32_t segment = 0; segment < segments; ++segment) {
            const std::uint32_t a = ring * (segments + 1) + segment;
            const std::uint32_t b = a + segments + 1;
            indices.insert(indices.end(), {a, b, a + 1, a + 1, b, b + 1});
        }
    }
    return positions;
}

std::vector<Core::float3> makeDisk(const float radius, const float innerRadius, const float height, std::vector<std::uint32_t>& indices) {
    constexpr std::uint32_t segments = 128;
    std::vector<Core::float3> positions;
    for (std::uint32_t segment = 0; segment < segments; ++segment) {
        const float phi = 2.0f * Core::pi * static_cast<float>(segment) / segments;
        positions.push_back({radius * std::cos(phi), radius * std::sin(phi), height});
        positions.push_back({innerRadius * std::cos(phi), innerRadius * std::sin(phi), height});
    }
    for (std::uint32_t segment = 0; segment < segments; ++segment) {
        const std::uint32_t next = (segment + 1) % segments;
        indices.insert(indices.end(), {segment * 2, next * 2, segment * 2 + 1, segment * 2 + 1, next * 2, next * 2 + 1});
    }
    return positions;
}

class Importer {
public:
    Importer(Scene::Scene& scene) : scene(scene), textureCache(scene) {
        Scene::Material fallback;
        fallback.name = "pbrt default";
        fallback.baseColor = {0.5f, 0.5f, 0.5f};
        fallback.roughness = 1.0f;
        fallback.specularIorLevel = 0.0f;
        materials.push_back(std::move(fallback));
    }

    std::expected<void, Core::Error> run(const std::vector<Command>& commands) {
        State state;
        std::vector<State> stack;
        std::unordered_map<std::string, Core::Transform> coordinateSystems;
        std::unordered_map<std::string, std::int32_t> namedMaterials;
        std::unordered_map<std::string, std::vector<ShapeRecord>> objects;
        std::vector<ShapeRecord>* target = &shapes;
        std::string currentObject;
        std::optional<CameraRecord> camera;
        const Command* film = nullptr;

        for (const Command& command : commands) {
            const std::string& name = command.name;
            const std::optional<std::vector<float>> arguments = parseNumbers(command.arguments);
            if (name == "WorldBegin") {
                coordinateSystems["world"] = {};
                state.transform = {};
            } else if (name == "AttributeBegin" || name == "TransformBegin") {
                stack.push_back(state);
            } else if (name == "AttributeEnd" || name == "TransformEnd") {
                if (stack.empty())
                    return failAt(command, std::format("unmatched {}", name));
                const State previous = stack.back();
                stack.pop_back();
                if (name == "TransformEnd")
                    state.transform = previous.transform;
                else
                    state = previous;
            } else if (name == "Identity") {
                state.transform = {};
            } else if (name == "Translate" || name == "Scale" || name == "Rotate" || name == "LookAt" || name == "Transform" || name == "ConcatTransform") {
                if (!arguments)
                    return failAt(command, "a transform has a value that is not a number");
                const std::vector<float>& a = *arguments;
                Core::Transform transform;
                if (name == "Translate")
                    transform = fromColumns({1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {a[0], a[1], a[2]});
                if (name == "Scale")
                    transform = fromColumns({a[0], 0.0f, 0.0f}, {0.0f, a[1], 0.0f}, {0.0f, 0.0f, a[2]}, {0.0f, 0.0f, 0.0f});
                if (name == "Rotate")
                    transform = rotation(a[0], {a[1], a[2], a[3]});
                if (name == "LookAt") {
                    //! pbrt's LookAt in its left-handed convention: right = up x dir.
                    const Core::float3 eye{a[0], a[1], a[2]};
                    const Core::float3 direction = Core::normalize(Core::float3{a[3], a[4], a[5]} - eye);
                    const Core::float3 right = Core::normalize(Core::cross(Core::normalize(Core::float3{a[6], a[7], a[8]}), direction));
                    transform = fromColumns(right, Core::cross(direction, right), direction, eye).inverse();
                }
                if (name == "Transform" || name == "ConcatTransform")
                    for (int row = 0; row < 3; ++row)
                        for (int column = 0; column < 4; ++column)
                            transform.rows[row][column] = a[column * 4 + row];
                state.transform = name == "Transform" ? transform : state.transform * transform;
            } else if (name == "CoordinateSystem") {
                coordinateSystems[command.arguments.front()] = state.transform;
            } else if (name == "CoordSysTransform") {
                if (const auto found = coordinateSystems.find(command.arguments.front()); found != coordinateSystems.end())
                    state.transform = found->second;
            } else if (name == "Film") {
                film = &command;
            } else if (name == "Camera") {
                camera = CameraRecord{&command, state.transform};
                coordinateSystems["camera"] = state.transform.inverse();
            } else if (name == "Texture") {
                addTexture(command);
            } else if (name == "Material") {
                state.material = addMaterial(command, command.arguments.front(), namedMaterials);
            } else if (name == "MakeNamedMaterial") {
                namedMaterials[command.arguments.front()] = addMaterial(command, command.getString("type", "diffuse"), namedMaterials);
            } else if (name == "NamedMaterial") {
                if (const auto found = namedMaterials.find(command.arguments.front()); found != namedMaterials.end())
                    state.material = found->second;
                else
                    Core::Log::warning("{}:{}: the material '{}' is not defined", command.source.string(), command.line, command.arguments.front());
            } else if (name == "AreaLightSource") {
                const Core::float3 radiance = getColor(command, "L").value_or(Core::float3{1.0f, 1.0f, 1.0f});
                state.emission = radiance * command.getFloat("scale", 1.0f);
            } else if (name == "ReverseOrientation") {
                state.reverse = !state.reverse;
            } else if (name == "Shape") {
                target->push_back({&command, state.transform, state.material, state.emission, state.reverse});
            } else if (name == "ObjectBegin") {
                if (!currentObject.empty())
                    return failAt(command, "ObjectBegin cannot be nested");
                currentObject = command.arguments.front();
                stack.push_back(state);
                target = &objects[currentObject];
            } else if (name == "ObjectEnd") {
                if (currentObject.empty() || stack.empty())
                    return failAt(command, "ObjectEnd has no ObjectBegin");
                currentObject.clear();
                target = &shapes;
                state = stack.back();
                stack.pop_back();
            } else if (name == "ObjectInstance") {
                const auto found = objects.find(command.arguments.front());
                if (found == objects.end()) {
                    Core::Log::warning("{}:{}: the object '{}' is not defined", command.source.string(), command.line, command.arguments.front());
                    continue;
                }
                for (ShapeRecord& shape : found->second)
                    if (std::expected<void, Core::Error> added = addShape(shape, state.transform); !added)
                        return added;
            } else if (name == "LightSource") {
                addLight(command, state.transform);
            }
        }
        for (const ShapeRecord& shape : shapes)
            if (std::expected<void, Core::Error> added = addShape(shape, {}); !added)
                return added;
        if (camera)
            addCamera(*camera, film);
        return {};
    }

private:
    static std::unexpected<Core::Error> failAt(const Command& command, const std::string_view message) {
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("{}:{}: {}", command.source.string(), command.line, message));
    }

    void addTexture(const Command& command) {
        if (command.arguments.size() < 3 || command.arguments[2] != "imagemap") {
            Core::Log::warning("{}:{}: only imagemap textures are imported", command.source.string(), command.line);
            return;
        }
        TextureDefinition definition;
        definition.path = (command.source.parent_path() / command.getString("filename")).lexically_normal();
        definition.linear = command.getString("encoding", "sRGB") == "linear";
        const float uscale = command.getFloat("uscale", 1.0f);
        const float vscale = command.getFloat("vscale", 1.0f);
        definition.transform.row0 = {uscale, 0.0f, command.getFloat("udelta", 0.0f)};
        definition.transform.row1 = {0.0f, vscale, command.getFloat("vdelta", 0.0f)};
        textures[command.arguments[0]] = std::move(definition);
    }

    std::optional<Scene::MaterialTexture> resolveTexture(const Command& command, const std::string_view name, const bool color) {
        const Parameter* parameter = command.find(name);
        if (!parameter || parameter->type != "texture" || parameter->values.empty())
            return std::nullopt;
        const auto found = textures.find(parameter->values.front());
        if (found == textures.end())
            return std::nullopt;
        const Materials::ColorSpace colorSpace = color && !found->second.linear ? Materials::ColorSpace::Srgb : Materials::ColorSpace::NonColor;
        return loadTexture(found->second.path, colorSpace, found->second.transform);
    }

    std::optional<Scene::MaterialTexture> loadTexture(const std::filesystem::path& path, const Materials::ColorSpace colorSpace,
                                                      const Scene::UvTransform& transform = {}) {
        std::expected<Scene::TextureHandle, Core::Error> handle = textureCache.load(path, colorSpace);
        if (!handle) {
            Core::Log::warning("pbrt texture '{}' is skipped: {}", path.string(), handle.error().message);
            return std::nullopt;
        }
        Scene::MaterialTexture texture;
        texture.texture = *handle;
        texture.transform = transform;
        return texture;
    }

    static float mapRoughness(const Command& command, const std::string& prefix, const float fallback) {
        const float roughness = command.getFloat(
            prefix + "roughness", 0.5f * (command.getFloat(prefix + "uroughness", fallback) + command.getFloat(prefix + "vroughness", fallback)));
        const Parameter* remap = command.find("remaproughness");
        const bool remapped = !remap || remap->values.empty() || remap->values.front() != "false";
        //! pbrt's alpha is sqrt(roughness) when remapped and roughness otherwise; Blender's alpha is roughness squared.
        return remapped ? std::pow(std::max(roughness, 0.0f), 0.25f) : std::sqrt(std::max(roughness, 0.0f));
    }

    void applyConductor(const Command& command, const std::string& prefix, Scene::Material& material) {
        material.metallic = 1.0f;
        material.roughness = mapRoughness(command, prefix, 0.0f);
        if (const std::optional<Core::float3> reflectance = getColor(command, prefix + "reflectance")) {
            material.baseColor = *reflectance;
            material.texture(Scene::TextureSlot::BaseColor) = resolveTexture(command, prefix + "reflectance", true);
            if (material.texture(Scene::TextureSlot::BaseColor))
                material.baseColor = {1.0f, 1.0f, 1.0f};
            return;
        }
        const Parameter* eta = command.find(prefix + "eta");
        if (eta && eta->type == "spectrum" && !eta->values.empty()) {
            material.baseColor = getNamedConductor(eta->values.front()).value_or(Core::float3{0.95f, 0.64f, 0.54f});
            return;
        }
        const std::optional<Core::float3> etaColor = getColor(command, prefix + "eta");
        const std::optional<Core::float3> k = getColor(command, prefix + "k");
        material.baseColor = etaColor && k ? fresnelConductor(*etaColor, *k) : Core::float3{0.95f, 0.64f, 0.54f};
    }

    std::int32_t addMaterial(const Command& command, const std::string& type, const std::unordered_map<std::string, std::int32_t>& named) {
        Scene::Material material;
        material.name =
            command.name == "MakeNamedMaterial" ? command.arguments.front() : std::format("{}:{}", command.source.filename().string(), command.line);
        material.roughness = 1.0f;
        material.specularIorLevel = 0.0f;
        const auto applyDiffuse = [&] {
            material.baseColor = getColor(command, "reflectance").or_else([&] { return getColor(command, "Kd"); }).value_or(Core::float3{0.5f, 0.5f, 0.5f});
            material.texture(Scene::TextureSlot::BaseColor) = resolveTexture(command, "reflectance", true);
            if (material.texture(Scene::TextureSlot::BaseColor))
                material.baseColor = {1.0f, 1.0f, 1.0f};
        };
        if (type == "diffuse" || type == "diffusetransmission") {
            applyDiffuse();
        } else if (type == "coateddiffuse") {
            applyDiffuse();
            material.coat = 1.0f;
            material.coatRoughness = mapRoughness(command, "", 0.0f);
        } else if (type == "conductor") {
            material.specularIorLevel = 0.5f;
            applyConductor(command, "", material);
        } else if (type == "coatedconductor") {
            material.specularIorLevel = 0.5f;
            applyConductor(command, "conductor.", material);
            material.coat = 1.0f;
            material.coatRoughness = mapRoughness(command, "interface.", 0.0f);
        } else if (type == "dielectric" || type == "thindielectric") {
            material.baseColor = {1.0f, 1.0f, 1.0f};
            material.transmission = 1.0f;
            material.specularIorLevel = 0.5f;
            material.roughness = mapRoughness(command, "", 0.0f);
            const Parameter* eta = command.find("eta");
            if (eta && eta->type == "spectrum" && !eta->values.empty()) {
                const NamedGlass glass = getNamedGlass(eta->values.front()).value_or(NamedGlass{});
                material.ior = glass.ior;
                material.dispersion = glass.abbe > 0.0f ? 20.0f / glass.abbe : 0.0f;
            } else {
                material.ior = command.getFloat("eta", 1.5f);
            }
        } else if (type == "interface") {
            material.alpha = 0.0f;
            material.alphaMode = Scene::AlphaMode::Mask;
        } else if (type == "mix") {
            const Parameter* parts = command.find("materials");
            if (parts && !parts->values.empty())
                if (const auto found = named.find(parts->values.front()); found != named.end())
                    return found->second;
            Core::Log::warning("{}:{}: mix materials use their first material", command.source.string(), command.line);
        } else {
            Core::Log::warning("{}:{}: pbrt material '{}' is imported as diffuse", command.source.string(), command.line, type);
            applyDiffuse();
        }
        if (const std::string normalMap = command.getString("normalmap"); !normalMap.empty())
            material.texture(Scene::TextureSlot::Normal) =
                loadTexture((command.source.parent_path() / normalMap).lexically_normal(), Materials::ColorSpace::NonColor);
        materials.push_back(std::move(material));
        return static_cast<std::int32_t>(materials.size() - 1);
    }

    Scene::MaterialHandle getMaterial(const std::int32_t index, const std::optional<Core::float3>& emission, const Command& shape) {
        const Core::float3 radiance = emission.value_or(Core::float3{});
        const std::string alpha = shape.find("alpha") ? shape.find("alpha")->values.front() : std::string();
        const auto key = std::make_tuple(index, radiance.x, radiance.y, radiance.z, alpha);
        if (const auto found = materialHandles.find(key); found != materialHandles.end())
            return found->second;
        Scene::Material material = materials[index];
        if (Core::maxComponent(radiance) > 0.0f) {
            material.emissionColor = radiance;
            material.emissionStrength = 1.0f;
        }
        if (!alpha.empty()) {
            material.texture(Scene::TextureSlot::Alpha) = resolveTexture(shape, "alpha", false);
            material.alpha = material.texture(Scene::TextureSlot::Alpha) ? 1.0f : shape.getFloat("alpha", 1.0f);
            material.alphaMode = Scene::AlphaMode::Blend;
        }
        const Scene::MaterialHandle handle = scene.addMaterial(std::move(material));
        materialHandles.emplace(key, handle);
        return handle;
    }

    std::expected<std::optional<MeshData>, Core::Error> readShape(const Command& command) {
        const std::string& type = command.arguments.front();
        MeshData data;
        data.name = std::format("{}:{}", command.source.filename().string(), command.line);
        if (type == "trianglemesh" || type == "bilinearmesh") {
            const std::vector<float> positions = command.find("P") ? command.find("P")->getFloats() : std::vector<float>{};
            std::vector<int> indices = command.find("indices") ? command.find("indices")->getInts() : std::vector<int>{};
            if (indices.empty() && positions.size() == (type == "trianglemesh" ? 9u : 12u))
                indices = type == "trianglemesh" ? std::vector<int>{0, 1, 2} : std::vector<int>{0, 1, 2, 3};
            const std::size_t corners = type == "trianglemesh" ? 3 : 4;
            if (positions.size() % 3 != 0 || indices.size() % corners != 0)
                return failAt(command, "the mesh has invalid array lengths");
            const std::size_t count = positions.size() / 3;
            for (std::size_t vertex = 0; vertex < count; ++vertex)
                data.positions.push_back({positions[vertex * 3], positions[vertex * 3 + 1], positions[vertex * 3 + 2]});
            for (std::size_t first = 0; first < indices.size(); first += corners) {
                for (std::size_t corner = 0; corner < corners; ++corner)
                    if (indices[first + corner] < 0 || static_cast<std::size_t>(indices[first + corner]) >= count)
                        return failAt(command, "the mesh has an index past its vertices");
                const auto at = [&](const std::size_t corner) { return static_cast<std::uint32_t>(indices[first + corner]); };
                if (corners == 3)
                    data.indices.insert(data.indices.end(), {at(0), at(1), at(2)});
                else
                    data.indices.insert(data.indices.end(), {at(0), at(1), at(3), at(0), at(3), at(2)});
            }
            if (const Parameter* normals = command.find("N")) {
                const std::vector<float> values = normals->getFloats();
                if (values.size() == positions.size())
                    for (std::size_t vertex = 0; vertex < count; ++vertex)
                        data.normals.push_back({values[vertex * 3], values[vertex * 3 + 1], values[vertex * 3 + 2]});
            }
            const Parameter* uv = command.find("uv") ? command.find("uv") : command.find("st");
            if (uv) {
                const std::vector<float> values = uv->getFloats();
                if (values.size() == count * 2)
                    for (std::size_t vertex = 0; vertex < count; ++vertex)
                        data.uvs[0].push_back({values[vertex * 2], values[vertex * 2 + 1]});
            }
        } else if (type == "plymesh") {
            const std::filesystem::path path = (command.source.parent_path() / command.getString("filename")).lexically_normal();
            if (path.extension() == ".gz") {
                Core::Log::warning("{}:{}: compressed PLY files are not supported; '{}' is skipped", command.source.string(), command.line, path.string());
                return std::nullopt;
            }
            Core::MappedFile file;
            if (std::expected<void, Core::Error> opened = file.open(path); !opened)
                return std::unexpected(std::move(opened.error()));
            PlyFile ply;
            if (std::expected<void, Core::Error> opened = ply.open(file.getBytes()); !opened)
                return Core::fail(opened.error().code, std::format("{}: {}", path.string(), opened.error().message));
            const PlyElement* vertex = ply.findElement("vertex");
            const PlyElement* face = ply.findElement("face");
            if (!vertex || !face)
                return failAt(command, std::format("'{}' has no vertex or face element", path.string()));
            static constexpr std::array<std::string_view, 3> positionNames{"x", "y", "z"};
            const std::vector<float> positions = ply.readFloats(*vertex, positionNames);
            for (std::size_t index = 0; index < vertex->count; ++index)
                data.positions.push_back({positions[index * 3], positions[index * 3 + 1], positions[index * 3 + 2]});
            if (vertex->findProperty("nx") != PlyFile::missing) {
                static constexpr std::array<std::string_view, 3> normalNames{"nx", "ny", "nz"};
                const std::vector<float> normals = ply.readFloats(*vertex, normalNames);
                for (std::size_t index = 0; index < vertex->count; ++index)
                    data.normals.push_back({normals[index * 3], normals[index * 3 + 1], normals[index * 3 + 2]});
            }
            for (const std::array<std::string_view, 2> names : {std::array<std::string_view, 2>{"u", "v"}, std::array<std::string_view, 2>{"s", "t"},
                                                                std::array<std::string_view, 2>{"texture_u", "texture_v"}}) {
                if (vertex->findProperty(names[0]) == PlyFile::missing)
                    continue;
                const std::vector<float> uvs = ply.readFloats(*vertex, names);
                for (std::size_t index = 0; index < vertex->count; ++index)
                    data.uvs[0].push_back({uvs[index * 2], uvs[index * 2 + 1]});
                break;
            }
            std::expected<std::vector<std::uint32_t>, Core::Error> triangles = ply.readTriangles(*face);
            if (!triangles)
                return std::unexpected(std::move(triangles.error()));
            if (std::ranges::any_of(*triangles, [&](const std::uint32_t index) { return index >= vertex->count; }))
                return failAt(command, std::format("'{}' has an index past its vertices", path.string()));
            data.indices = std::move(*triangles);
        } else if (type == "sphere") {
            data.positions = makeSphere(command.getFloat("radius", 1.0f), data.indices, data.uvs[0]);
            for (const Core::float3 position : data.positions)
                data.normals.push_back(Core::normalize(position));
        } else if (type == "disk") {
            data.positions = makeDisk(command.getFloat("radius", 1.0f), command.getFloat("innerradius", 0.0f), command.getFloat("height", 0.0f), data.indices);
            data.normals.assign(data.positions.size(), Core::float3{0.0f, 0.0f, 1.0f});
        } else {
            Core::Log::warning("{}:{}: pbrt shape '{}' is not supported and is skipped", command.source.string(), command.line, type);
            return std::nullopt;
        }
        return data;
    }

    std::expected<void, Core::Error> addShape(const ShapeRecord& shape, const Core::Transform& instance) {
        const Command& command = *shape.command;
        const auto key = std::make_tuple(&command, shape.material, shape.reverse, shape.emission.has_value());
        auto found = meshes.find(key);
        if (found == meshes.end()) {
            std::expected<std::optional<MeshData>, Core::Error> read = readShape(command);
            if (!read)
                return std::unexpected(std::move(read.error()));
            if (!*read) {
                meshes.emplace(key, std::nullopt);
                return {};
            }
            MeshData& data = **read;
            if (shape.reverse) {
                for (std::size_t first = 0; first + 2 < data.indices.size(); first += 3)
                    std::swap(data.indices[first + 1], data.indices[first + 2]);
                for (Core::float3& normal : data.normals)
                    normal = -normal;
            }
            data.appendSubset(getMaterial(shape.material, shape.emission, command), 0, static_cast<std::uint32_t>(data.indices.size()));
            found = meshes.emplace(key, scene.addMesh(MeshBuilder::build(std::move(data)))).first;
        }
        if (!found->second)
            return {};
        const Core::Transform transform = instance * shape.transform;
        const Scene::InstanceHandle added =
            scene.addInstance({transform, transform, *found->second, Scene::InstanceFlags::None, static_cast<std::uint32_t>(command.line)});
        scene.addNode({scene.find(*found->second)->name, transform, {}, added});
        return {};
    }

    void addLightNode(Scene::Light light) {
        const Core::Transform transform = light.transform;
        std::string name = light.name;
        scene.addNode({std::move(name), transform, {}, scene.addLight(std::move(light))});
    }

    void addLight(const Command& command, const Core::Transform& transform) {
        const std::string& type = command.arguments.front();
        const float scale = command.getFloat("scale", 1.0f);
        Scene::Light light;
        light.name = std::format("{} light", type);
        const Core::float3 from = getVector(command, "from", {0.0f, 0.0f, 0.0f});
        const Core::float3 to = getVector(command, "to", {0.0f, 0.0f, 1.0f});
        if (type == "point" || type == "spot") {
            const Core::float3 intensity = getColor(command, "I").value_or(Core::float3{1.0f, 1.0f, 1.0f}) * scale;
            const float peak = std::max(Core::maxComponent(intensity), 1e-12f);
            light.color = intensity / peak;
            light.power = 4.0f * Core::pi * peak;
            const Core::float3 position = transform.transformPoint(from);
            if (type == "point") {
                light.type = Scene::LightType::Point;
                light.transform = fromColumns({1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, position);
                light.radius = command.getFloat("radius", 0.0f);
                if (const float power = command.getFloat("power", -1.0f); power > 0.0f)
                    light.power = power;
            } else {
                light.type = Scene::LightType::Spot;
                light.transform = facing(position, transform.transformVector(to - from));
                const float cone = command.getFloat("coneangle", 30.0f) * Core::pi / 180.0f;
                const float delta = command.getFloat("conedeltaangle", 5.0f) * Core::pi / 180.0f;
                light.spotAngle = 2.0f * cone;
                light.spotBlend = cone > 0.0f ? std::clamp(delta / cone, 0.0f, 1.0f) : 0.0f;
                if (const float power = command.getFloat("power", -1.0f); power > 0.0f) {
                    const float solidAngle = 2.0f * Core::pi * (1.0f - 0.5f * (std::cos(cone) + std::cos(cone - delta)));
                    light.power = 4.0f * Core::pi * power / std::max(solidAngle, 1e-6f);
                }
            }
            addLightNode(std::move(light));
            return;
        }
        if (type == "distant") {
            const Core::float3 radiance = getColor(command, "L").value_or(Core::float3{1.0f, 1.0f, 1.0f}) * scale;
            const float peak = std::max(Core::maxComponent(radiance), 1e-12f);
            light.type = Scene::LightType::Directional;
            light.color = radiance / peak;
            light.power = peak;
            if (const float illuminance = command.getFloat("illuminance", -1.0f); illuminance > 0.0f)
                light.power = illuminance / lumensPerWatt;
            light.transform = facing({0.0f, 0.0f, 0.0f}, transform.transformVector(to - from));
            addLightNode(std::move(light));
            return;
        }
        if (type == "infinite") {
            addEnvironment(command, transform, scale);
            return;
        }
        Core::Log::warning("{}:{}: pbrt light '{}' is not supported and is skipped", command.source.string(), command.line, type);
    }

    void addEnvironment(const Command& command, const Core::Transform& transform, const float scale) {
        Scene::Environment environment;
        environment.strength = scale;
        environment.color = getColor(command, "L").value_or(Core::float3{1.0f, 1.0f, 1.0f});
        const std::string filename = command.getString("filename");
        if (!filename.empty()) {
            const std::filesystem::path path = (command.source.parent_path() / filename).lexically_normal();
            std::expected<Scene::TextureHandle, Core::Error> texture = convertEqualArea(path, transform.inverse());
            if (texture) {
                environment.texture = *texture;
                environment.color = {1.0f, 1.0f, 1.0f};
            } else {
                Core::Log::warning("pbrt environment '{}' is skipped: {}", path.string(), texture.error().message);
            }
        }
        scene.setEnvironment(environment);
    }

    //! pbrt-v4 environment maps are equal-area octahedral squares in a +Z-up light space; this resamples one to the scene's equirect map.
    std::expected<Scene::TextureHandle, Core::Error> convertEqualArea(const std::filesystem::path& path, const Core::Transform& lightFromWorld) {
        Core::MappedFile file;
        if (std::expected<void, Core::Error> opened = file.open(path); !opened)
            return std::unexpected(std::move(opened.error()));
        std::expected<Scene::Texture, Core::Error> decoded = Materials::TextureCache::decode(path.string(), file.getBytes(), Materials::ColorSpace::Linear);
        if (!decoded)
            return std::unexpected(std::move(decoded.error()));
        const Scene::Texture& source = *decoded;
        if (source.width != source.height)
            return Core::fail(Core::ErrorCode::Unsupported, "the map is not square, so it is not an equal-area map");
        const std::uint32_t size = source.width;
        const auto texel = [&](std::uint32_t x, std::uint32_t y) {
            x = std::min(x, size - 1);
            y = std::min(y, size - 1);
            const std::size_t index = static_cast<std::size_t>(y) * size + x;
            if (source.format == NoorRHI::ImageFormat::Rgba16Float) {
                const auto* halves = reinterpret_cast<const NoorRHI::half*>(source.data.data());
                return Core::float3{Core::fromHalf(halves[index * 4]), Core::fromHalf(halves[index * 4 + 1]), Core::fromHalf(halves[index * 4 + 2])};
            }
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(source.data.data());
            const auto linear = [](const std::uint8_t value) {
                const float c = value / 255.0f;
                return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
            };
            const bool srgb = source.format == NoorRHI::ImageFormat::Rgba8Srgb;
            const auto channel = [&](const std::size_t offset) { return srgb ? linear(bytes[index * 4 + offset]) : bytes[index * 4 + offset] / 255.0f; };
            return Core::float3{channel(0), channel(1), channel(2)};
        };
        const std::uint32_t width = size * 2;
        const std::uint32_t height = size;
        std::vector<float> pixels(static_cast<std::size_t>(width) * height * 4);
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) {
                const Core::float3 world = Scene::equirectToDirection({(x + 0.5f) / width, (y + 0.5f) / height});
                const Core::float3 d = Core::normalize(lightFromWorld.transformVector(world));
                const float ax = std::abs(d.x);
                const float ay = std::abs(d.y);
                const float r = std::sqrt(std::max(1.0f - std::abs(d.z), 0.0f));
                const float a = std::max(ax, ay);
                const float b = a == 0.0f ? 0.0f : std::min(ax, ay) / a;
                float phi = std::atan(b) * 2.0f / Core::pi;
                if (ax < ay)
                    phi = 1.0f - phi;
                float v = phi * r;
                float u = r - v;
                if (d.z < 0.0f) {
                    std::swap(u, v);
                    u = 1.0f - u;
                    v = 1.0f - v;
                }
                u = std::copysign(u, d.x);
                v = std::copysign(v, d.y);
                const float sx = 0.5f * (u + 1.0f) * size - 0.5f;
                const float sy = 0.5f * (v + 1.0f) * size - 0.5f;
                const std::uint32_t x0 = static_cast<std::uint32_t>(std::clamp(std::floor(sx), 0.0f, static_cast<float>(size - 1)));
                const std::uint32_t y0 = static_cast<std::uint32_t>(std::clamp(std::floor(sy), 0.0f, static_cast<float>(size - 1)));
                const float fx = std::clamp(sx - static_cast<float>(x0), 0.0f, 1.0f);
                const float fy = std::clamp(sy - static_cast<float>(y0), 0.0f, 1.0f);
                const Core::float3 color =
                    (texel(x0, y0) * (1.0f - fx) + texel(x0 + 1, y0) * fx) * (1.0f - fy) + (texel(x0, y0 + 1) * (1.0f - fx) + texel(x0 + 1, y0 + 1) * fx) * fy;
                float* pixel = pixels.data() + (static_cast<std::size_t>(y) * width + x) * 4;
                pixel[0] = color.x;
                pixel[1] = color.y;
                pixel[2] = color.z;
                pixel[3] = 1.0f;
            }
        }
        return scene.addTexture(Materials::TextureCache::fromRgbaFloat(path.string(), width, height, pixels));
    }

    static std::vector<Scene::LensElement> readLensFile(const std::filesystem::path& path) {
        std::vector<Scene::LensElement> elements;
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line.front() == '#')
                continue;
            std::istringstream stream(line);
            float radius = 0.0f;
            float thickness = 0.0f;
            float ior = 0.0f;
            float aperture = 0.0f;
            if (!(stream >> radius >> thickness >> ior >> aperture))
                continue;
            elements.push_back({radius * 0.001f, thickness * 0.001f, ior == 0.0f ? 1.0f : ior, 0.0f, aperture * 0.0005f});
        }
        return elements;
    }

    void addCamera(const CameraRecord& record, const Command* film) {
        const Command& command = *record.command;
        const float width = film ? film->getFloat("xresolution", 1280.0f) : 1280.0f;
        const float height = film ? film->getFloat("yresolution", 720.0f) : 720.0f;
        const float aspect = width / height;
        Scene::Camera camera;
        camera.name = "pbrt camera";
        //! pbrt cameras look down +Z; ours look down -Z with the same image x and y axes.
        Core::Transform worldFromCamera = record.cameraFromWorld.inverse();
        for (int row = 0; row < 3; ++row)
            worldFromCamera.rows[row][2] = -worldFromCamera.rows[row][2];
        camera.transform = worldFromCamera;
        camera.previousTransform = worldFromCamera;
        camera.sensor.width = camera.sensor.height * aspect;
        const std::string& type = command.arguments.front();
        if (type == "orthographic") {
            camera.type = Scene::CameraType::Orthographic;
            camera.orthographicHeight = aspect >= 1.0f ? 2.0f : 2.0f / aspect;
        } else if (type == "realistic") {
            camera.type = Scene::CameraType::Realistic;
            const float diagonal = (film ? film->getFloat("diagonal", 35.0f) : 35.0f) * 0.001f;
            camera.sensor.width = diagonal * aspect / std::sqrt(aspect * aspect + 1.0f);
            camera.sensor.height = camera.sensor.width / aspect;
            camera.lens.elements = readLensFile((command.source.parent_path() / command.getString("lensfile")).lexically_normal());
            const float stop = command.getFloat("aperturediameter", 1.0f) * 0.0005f;
            for (Scene::LensElement& element : camera.lens.elements)
                if (element.curvatureRadius == 0.0f)
                    element.apertureRadius = std::min(element.apertureRadius, stop);
            camera.focusDistance = command.getFloat("focusdistance", 10.0f);
            if (camera.lens.isEmpty())
                Core::Log::warning("{}:{}: the realistic camera has no readable lens file", command.source.string(), command.line);
        } else {
            if (type != "perspective")
                Core::Log::warning("{}:{}: pbrt camera '{}' is imported as perspective", command.source.string(), command.line, type);
            const float fov = command.getFloat("fov", 90.0f) * Core::pi / 180.0f;
            camera.verticalFov = aspect >= 1.0f ? fov : 2.0f * std::atan(std::tan(0.5f * fov) / aspect);
            camera.focusDistance = command.getFloat("focaldistance", 1e6f);
            const float lensRadius = command.getFloat("lensradius", 0.0f);
            if (lensRadius > 0.0f)
                camera.fStop = 0.5f * camera.getFocalLength() / lensRadius;
        }
        const Core::Transform transform = camera.transform;
        std::string name = camera.name;
        const Scene::CameraHandle added = scene.addCamera(std::move(camera));
        scene.addNode({std::move(name), transform, {}, added});
        scene.setActiveCamera(added);
    }

    Scene::Scene& scene;
    Materials::TextureCache textureCache;
    std::vector<Scene::Material> materials;
    std::unordered_map<std::string, TextureDefinition> textures;
    std::vector<ShapeRecord> shapes;
    std::map<std::tuple<std::int32_t, float, float, float, std::string>, Scene::MaterialHandle> materialHandles;
    std::map<std::tuple<const Command*, std::int32_t, bool, bool>, std::optional<Scene::MeshHandle>> meshes;
};

}

std::expected<void, Core::Error> PbrtImporter::import(const std::filesystem::path& path, Scene::Scene& scene) {
    std::expected<std::vector<Command>, Core::Error> commands = Pbrt::Parser::parseFile(path);
    if (!commands)
        return std::unexpected(std::move(commands.error()));
    return Importer(scene).run(*commands);
}

}
