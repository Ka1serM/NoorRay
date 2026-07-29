#define GLM_ENABLE_EXPERIMENTAL
#include "Scene/SceneImporter.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/quaternion.hpp>

#include "Camera/CameraInstance.h"
#include "Camera/RealisticCamera.h"
#include "Camera/HybridPsfCamera.h"
#include "Camera/ThinLensCamera.h"
#include "Log.h"
#include "MaterialX/MaterialXCompiler.h"
#include "Mesh/Assets/MeshAsset.h"
#include "Mesh/Assets/PlyMeshLoader.h"
#include "Mesh/Transform.h"
#include "Shading/Sellmeier.h"
#include "Scene/LightInstance.h"
#include "Scene/MeshInstance.h"
#include "Scene/PbrtParser.h"
#include "Scene/Scene.h"
#include "Scene/Texture.h"
#include "libross/imaging/cameralens/lenssystemio/CameraLensSystemReader.h"
#include "libross/imaging/cameralens/raytracing/hyperfocaldistance/HyperFocalDistanceDeterminator.h"
#include "libross/imaging/imagesensor/ImageSensorReader.h"
#include "openlensfileio/glasscatalogs/glasscatalog/GlassCatalogLibrary.h"

namespace {
using nr::pbrt::Command;
using nr::pbrt::Parameter;

std::filesystem::path resolvePbrtPath(const std::string& filepath)
{
    const std::filesystem::path direct(filepath);
    if (std::filesystem::exists(direct)) return direct;
    const std::filesystem::path fallback = std::filesystem::path(NOORRAY_ASSET_DIR) / filepath;
    return std::filesystem::exists(fallback) ? fallback : direct;
}

std::string relativeAssetPath(const Command& command, const std::string_view parameter)
{
    const Parameter* parsed = command.find(parameter);
    const std::optional<std::string> parsedValue = parsed ? parsed->stringValue() : std::nullopt;
    const std::string value = parsedValue.value_or("");
    if (value.empty()) return {};
    const std::filesystem::path path(value);
    return (path.is_absolute() ? path : command.source.parent_path() / path).lexically_normal().string();
}

std::string relativeAssetList(const Command& command, const std::string_view parameter)
{
    const Parameter* parsed = command.find(parameter);
    const std::optional<std::string> parsedValue = parsed ? parsed->stringValue() : std::nullopt;
    std::string list = parsedValue.value_or("");
    std::ranges::replace(list, ',', ';');
    std::string result;
    size_t begin = 0;
    while (begin <= list.size()) {
        const size_t end = list.find(';', begin);
        std::string item = list.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        const size_t first = item.find_first_not_of(" \t");
        const size_t last = item.find_last_not_of(" \t");
        if (first != std::string::npos) {
            std::filesystem::path path = item.substr(first, last - first + 1);
            if (!path.is_absolute()) path = command.source.parent_path() / path;
            if (!result.empty()) result += ';';
            result += path.lexically_normal().string();
        }
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return result;
}

float number(const std::string& text, const Command& command)
{
    try {
        size_t consumed = 0;
        const float result = std::stof(text, &consumed);
        if (consumed != text.size()) throw std::invalid_argument("suffix");
        return result;
    } catch (...) {
        throw std::runtime_error(command.source.string() + ":" + std::to_string(command.line)
            + ": expected a number, got '" + text + "'");
    }
}

float scalar(const Command& command, const std::string_view name, const float fallback)
{
    if (const Parameter* value = command.find(name))
        if (const auto result = value->floatValue()) return *result;
    return fallback;
}

bool boolean(const Command& command, const std::string_view name, const bool fallback)
{
    const Parameter* parameter = command.find(name);
    if (!parameter || parameter->values.empty()) return fallback;
    if (parameter->values.front() == "true") return true;
    if (parameter->values.front() == "false") return false;
    return fallback;
}

struct OpticalSettings {
    float apertureDiameterMm;
    float focusDistanceCm;
};

OpticalSettings opticalSettings(const Command& command, const std::string& lensPath,
    const std::string& sensorPath, const std::string& catalogPaths)
{
    OpticalSettings result{
        scalar(command, "aperturediameter", -1.f),
        scalar(command, "focusdistance", 5.f) * 100.f};
    const bool focusAtHyperfocal = boolean(command, "focusathyperfocal", false);
    if (result.apertureDiameterMm > 0.f && !focusAtHyperfocal)
        return result;

    olio::GlassCatalogLibrary catalogs;
    std::string catalogList = catalogPaths;
    std::ranges::replace(catalogList, ';', ',');
    if (!catalogList.empty()) catalogs.loadCatalogsFromCommaSeperatedString(catalogList);
    ross::CameraLens lens = ross::CameraLensSystemReader::readCameraLens(
        lensPath, catalogs, ross::ReadOptions{1.0f, false});
    if (result.apertureDiameterMm > 0.f)
        lens.changeAperture_mm(result.apertureDiameterMm);
    else
        lens.changeAperture(scalar(command, "fstop", 4.f));
    result.apertureDiameterMm = std::max(0.f, lens.getApertureRadius() * 20.f);

    if (focusAtHyperfocal) {
        if (sensorPath.empty())
            throw std::runtime_error(
                "sensorFilePath is required when focusAtHyperfocal is enabled");
        ross::ImageSensor sensor = ross::ImageSensorReader::readFile(sensorPath);
        ross::HyperFocalDistanceDeterminator determiner(lens, sensor);
        result.focusDistanceCm = determiner.determine().centimeter();
    }
    return result;
}

std::string stringValue(const Command& command, const std::string_view name,
    const std::string& fallback = {})
{
    if (const Parameter* value = command.find(name))
        if (const auto result = value->stringValue()) return *result;
    return fallback;
}

glm::vec3 rgb(const Command& command, const std::string_view name,
    const glm::vec3 fallback)
{
    if (const Parameter* value = command.find(name)) {
        const std::vector<float> values = value->floatValues();
        if (values.size() >= 3) return {values[0], values[1], values[2]};
        if (values.size() == 1) return glm::vec3(values[0]);
    }
    return fallback;
}

glm::vec3 point(const Command& command, const std::string_view name,
    const glm::vec3 fallback)
{
    return rgb(command, name, fallback);
}

glm::mat4 pbrtMatrix(const Command& command)
{
    glm::mat4 result(1.f);
    // PBRT serializes explicit transforms in its row-vector convention and
    // transposes them when building the active transform (see PBRT's
    // BasicSceneBuilder::Transform). GLM uses column vectors, so write the
    // PBRT rows into GLM's columns to perform that same transpose. This is
    // especially important for files such as VehicleRearview, whose
    // translations are stored in input elements 12..14 (the fourth row).
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            result[row][column] = number(command.arguments[row * 4 + column], command);
    return result;
}

glm::mat4 lookAtCameraFromWorld(const Command& command)
{
    const glm::vec3 eye(number(command.arguments[0], command), number(command.arguments[1], command),
        number(command.arguments[2], command));
    const glm::vec3 target(number(command.arguments[3], command), number(command.arguments[4], command),
        number(command.arguments[5], command));
    const glm::vec3 up = glm::normalize(glm::vec3(number(command.arguments[6], command),
        number(command.arguments[7], command), number(command.arguments[8], command)));
    const glm::vec3 forward = glm::normalize(target - eye);
    const glm::vec3 right = glm::normalize(glm::cross(up, forward));
    const glm::vec3 correctedUp = glm::cross(forward, right);
    const glm::mat4 worldFromCamera(
        glm::vec4(right, 0.f), glm::vec4(correctedUp, 0.f),
        glm::vec4(forward, 0.f), glm::vec4(eye, 1.f));
    return glm::inverse(worldFromCamera);
}

MaterialAuthoring makeMaterial(const Command& command,
    const std::unordered_map<std::string, int>& textures)
{
    MaterialAuthoring material{};
    const std::string type = command.arguments.empty() ? "diffuse" : command.arguments.front();
    if (type != "diffuse" && type != "coateddiffuse" && type != "conductor"
        && type != "coatedconductor" && type != "dielectric" && type != "thindielectric")
        throw std::runtime_error("PBRT material '" + type + "' is not supported by NoorRay at "
            + command.source.string() + ":" + std::to_string(command.line));
    material.albedo = rgb(command, "reflectance",
        rgb(command, "Kd", rgb(command, "base_color", glm::vec3(0.5f))));
    material.roughness = std::clamp(scalar(command, "roughness",
        scalar(command, "uroughness", type == "diffuse" ? 1.f : 0.2f)), 0.f, 1.f);
    material.metallic = std::clamp(scalar(command, "metallic", 0.f), 0.f, 1.f);
    if (type == "conductor" || type == "coatedconductor") material.metallic = 1.f;
    if (type == "dielectric" || type == "thindielectric") {
        material.albedo = glm::vec3(1.f);
        material.transmission = 1.f;
        material.sellmeier = constantIorSellmeier(scalar(command, "eta", 1.5f));
    }
    if (const Parameter* texture = command.find("reflectance")) {
        if (texture->type == "texture" && !texture->values.empty()) {
            const auto found = textures.find(texture->values.front());
            if (found != textures.end()) material.albedoIndex = found->second;
        }
    }
    return material;
}

struct ShapeRecord {
    Command command;
    glm::mat4 transform{1.f};
    MaterialAuthoring material{};
    bool reverse{};
};

struct State {
    glm::mat4 transform{1.f};
    MaterialAuthoring material{};
    std::string namedMaterial;
    glm::vec3 areaEmission{};
    bool reverse{};
};

void computeMissingNormals(std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices)
{
    for (Vertex& vertex : vertices) vertex.normal = glm::vec3(0.f);
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const glm::vec3 normal = glm::cross(vertices[indices[i + 1]].position - vertices[indices[i]].position,
            vertices[indices[i + 2]].position - vertices[indices[i]].position);
        vertices[indices[i]].normal += normal;
        vertices[indices[i + 1]].normal += normal;
        vertices[indices[i + 2]].normal += normal;
    }
    for (Vertex& vertex : vertices) {
        vertex.normal = glm::length2(vertex.normal) > 0.f ? glm::normalize(vertex.normal) : glm::vec3(0, 1, 0);
        const glm::vec3 helper = std::abs(vertex.normal.y) < 0.99f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        vertex.tangent = glm::normalize(glm::cross(helper, vertex.normal));
    }
}

void addShape(Scene& scene, const ShapeRecord& shape, const size_t index)
{
    if (shape.command.arguments.empty()) return;
    const std::string& type = shape.command.arguments.front();
    const std::string name = type + "_" + std::to_string(index);
    MaterialAuthoring material = shape.material;
    const MaterialX::DocumentPtr materialDocument =
        nr::materialx::documentFromAuthoring(material);
    glm::mat4 transform = shape.transform;
    MeshAssetRef meshAsset;

    if (type == "sphere") {
        const float radius = scalar(shape.command, "radius", 1.f);
        transform *= glm::scale(glm::mat4(1.f), glm::vec3(radius * 2.f));
        meshAsset = scene.add(MeshAsset::CreateSphere(scene, name, materialDocument));
    } else if (type == "disk") {
        const float radius = scalar(shape.command, "radius", 1.f);
        transform *= glm::rotate(glm::mat4(1.f), glm::half_pi<float>(), glm::vec3(1, 0, 0));
        transform *= glm::scale(glm::mat4(1.f), glm::vec3(radius * 2.f));
        meshAsset = scene.add(MeshAsset::CreateDisk(scene, name, materialDocument));
    } else if (type == "plymesh") {
        const std::string filename = relativeAssetPath(shape.command, "filename");
        if (filename.empty())
            throw std::runtime_error("plymesh requires a filename at "
                + shape.command.source.string() + ":" + std::to_string(shape.command.line));

        PlyMeshData mesh;
        try {
            mesh = PlyMeshLoader::Load(filename);
        } catch (const std::exception& error) {
            throw std::runtime_error("failed to load PBRT plymesh at "
                + shape.command.source.string() + ":" + std::to_string(shape.command.line)
                + ": " + error.what());
        }

        if (shape.reverse) {
            for (size_t i = 0; i < mesh.indices.size(); i += 3)
                std::swap(mesh.indices[i + 1], mesh.indices[i + 2]);
            for (Vertex& vertex : mesh.vertices)
                vertex.normal = -vertex.normal;
        }

        const std::string meshName = std::filesystem::path(filename).stem().string();
        meshAsset = scene.add(MeshAsset(scene,
            meshName.empty() ? name : meshName,
            mesh.vertices, mesh.indices,
            std::vector<Face>(mesh.indices.size() / 3, Face{0}),
            std::vector<MaterialX::DocumentPtr>{materialDocument}));
    } else if (type == "trianglemesh") {
        const Parameter* positions = shape.command.find("P");
        const Parameter* indexParameter = shape.command.find("indices");
        if (!positions || !indexParameter)
            throw std::runtime_error("trianglemesh requires P and indices at "
                + shape.command.source.string() + ":" + std::to_string(shape.command.line));
        const std::vector<float> p = positions->floatValues();
        const std::vector<int> sourceIndices = indexParameter->intValues();
        if (p.size() % 3 != 0 || sourceIndices.size() % 3 != 0)
            throw std::runtime_error("invalid trianglemesh array length at "
                + shape.command.source.string() + ":" + std::to_string(shape.command.line));
        std::vector<Vertex> vertices(p.size() / 3);
        for (size_t i = 0; i < vertices.size(); ++i) {
            vertices[i].position = {p[i * 3], p[i * 3 + 1], p[i * 3 + 2]};
            vertices[i].tangentSign = 1.f;
        }
        if (const Parameter* uvParameter = shape.command.find("uv")) {
            const auto uv = uvParameter->floatValues();
            for (size_t i = 0; i < vertices.size() && i * 2 + 1 < uv.size(); ++i)
                vertices[i].uv = {uv[i * 2], 1.f - uv[i * 2 + 1]};
        } else if (const Parameter* stParameter = shape.command.find("st")) {
            const auto uv = stParameter->floatValues();
            for (size_t i = 0; i < vertices.size() && i * 2 + 1 < uv.size(); ++i)
                vertices[i].uv = {uv[i * 2], 1.f - uv[i * 2 + 1]};
        }
        std::vector<uint32_t> indices;
        indices.reserve(sourceIndices.size());
        for (const int value : sourceIndices) {
            if (value < 0 || static_cast<size_t>(value) >= vertices.size())
                throw std::runtime_error("trianglemesh index out of bounds");
            indices.push_back(static_cast<uint32_t>(value));
        }
        if (shape.reverse)
            for (size_t i = 0; i < indices.size(); i += 3) std::swap(indices[i + 1], indices[i + 2]);
        if (const Parameter* normals = shape.command.find("N")) {
            const auto n = normals->floatValues();
            for (size_t i = 0; i < vertices.size() && i * 3 + 2 < n.size(); ++i) {
                vertices[i].normal = glm::normalize(glm::vec3(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]));
                if (shape.reverse) vertices[i].normal = -vertices[i].normal;
                const glm::vec3 helper = std::abs(vertices[i].normal.y) < .99f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
                vertices[i].tangent = glm::normalize(glm::cross(helper, vertices[i].normal));
            }
        } else computeMissingNormals(vertices, indices);
        meshAsset = scene.add(MeshAsset(scene, name, vertices, indices,
            std::vector<Face>(indices.size() / 3, Face{0}),
            std::vector<MaterialX::DocumentPtr>{materialDocument}));
    } else {
        LOG_ERROR("PBRT shape '" << type << "' is not supported; skipping "
            << shape.command.source.string() << ':' << shape.command.line);
        return;
    }

    auto instance = std::make_unique<MeshInstance>(scene, name, meshAsset, Transform(transform));
    instance->setSource("pbrt", shape.command.source.string());
    scene.add(std::move(instance));
}

glm::vec3 transformedPoint(const glm::mat4& transform, const glm::vec3 value)
{
    return glm::vec3(transform * glm::vec4(value, 1.f));
}

} // namespace

void SceneImporter::ImportPbrtScene(Scene& scene, const std::string& filepath)
{
    const std::filesystem::path filePath = resolvePbrtPath(filepath);
    if (!std::filesystem::exists(filePath))
        throw std::runtime_error("File not found: " + filepath);

    const std::vector<Command> commands = nr::pbrt::parseFile(filePath);
    State state;
    state.material.albedo = glm::vec3(0.5f);
    state.material.roughness = 1.f;
    std::vector<State> stack;
    std::unordered_map<std::string, glm::mat4> coordinateSystems;
    std::unordered_map<std::string, MaterialAuthoring> namedMaterials;
    std::unordered_map<std::string, int> textures;
    // Materials carry bare texture slot indices, so the importer holds a
    // reference per texture until the materials using them reach the scene.
    std::vector<TextureRef> importedTextures;
    std::unordered_map<std::string, std::vector<ShapeRecord>> objectDefinitions;
    std::vector<ShapeRecord> shapes;
    std::vector<ShapeRecord>* shapeTarget = &shapes;
    std::string currentObject;
    glm::mat4 cameraFromWorld(1.f);
    std::string cameraType = "perspective";
    float cameraFov = 90.f;
    float lensRadius = 0.f;
    float focalDistance = 5.f;
    uint32_t resolutionX = 1280, resolutionY = 720;
    bool hasCamera = false;
    std::optional<Command> cameraCommand;

    for (const Command& command : commands) {
        const std::string& name = command.name;
        if (name == "WorldBegin") {
            state.transform = glm::mat4(1.f);
        } else if (name == "AttributeBegin" || name == "TransformBegin") {
            stack.push_back(state);
        } else if (name == "AttributeEnd" || name == "TransformEnd") {
            if (stack.empty()) throw std::runtime_error("unmatched " + name + " in " + filepath);
            state = stack.back(); stack.pop_back();
        } else if (name == "Identity") state.transform = glm::mat4(1.f);
        else if (name == "Translate") state.transform *= glm::translate(glm::mat4(1.f), {
            number(command.arguments[0], command), number(command.arguments[1], command), number(command.arguments[2], command)});
        else if (name == "Scale") state.transform *= glm::scale(glm::mat4(1.f), {
            number(command.arguments[0], command), number(command.arguments[1], command), number(command.arguments[2], command)});
        else if (name == "Rotate") state.transform *= glm::rotate(glm::mat4(1.f),
            glm::radians(number(command.arguments[0], command)), glm::normalize(glm::vec3(
                number(command.arguments[1], command), number(command.arguments[2], command), number(command.arguments[3], command))));
        else if (name == "LookAt") state.transform *= lookAtCameraFromWorld(command);
        else if (name == "Transform") state.transform = pbrtMatrix(command);
        else if (name == "ConcatTransform") state.transform *= pbrtMatrix(command);
        else if (name == "CoordinateSystem") coordinateSystems[command.arguments.front()] = state.transform;
        else if (name == "CoordSysTransform") {
            if (const auto found = coordinateSystems.find(command.arguments.front()); found != coordinateSystems.end())
                state.transform = found->second;
        } else if (name == "Film") {
            resolutionX = static_cast<uint32_t>(std::max(1.f, scalar(command, "xresolution", 1280.f)));
            resolutionY = static_cast<uint32_t>(std::max(1.f, scalar(command, "yresolution", 720.f)));
        } else if (name == "Camera") {
            hasCamera = true;
            cameraFromWorld = state.transform;
            cameraType = command.arguments.front();
            cameraCommand = command;
            cameraFov = scalar(command, "fov", 90.f);
            lensRadius = scalar(command, "lensradius", 0.f);
            focalDistance = scalar(command, "focaldistance", 5.f);
        } else if (name == "Texture") {
            if (command.arguments.size() >= 3 && command.arguments[2] == "imagemap") {
                const std::string filename = stringValue(command, "filename");
                if (!filename.empty()) {
                    const std::filesystem::path texturePath =
                        (command.source.parent_path() / filename).lexically_normal();
                    if (!std::filesystem::exists(texturePath)) {
                        LOG_WARN("PBRT texture not found; skipping: " << texturePath.string()
                            << " (" << command.source.string() << ':' << command.line << ')');
                    } else {
                        importedTextures.push_back(scene.add(Texture(
                            texturePath.string(), TextureEncoding::Srgb8)));
                        textures[command.arguments[0]] =
                            static_cast<int>(importedTextures.back().index());
                    }
                }
            }
        } else if (name == "Material") {
            state.material = makeMaterial(command, textures);
            state.namedMaterial.clear();
        } else if (name == "MakeNamedMaterial") {
            Command materialCommand = command;
            materialCommand.arguments = {stringValue(command, "type", "diffuse")};
            namedMaterials[command.arguments.front()] = makeMaterial(materialCommand, textures);
        } else if (name == "NamedMaterial") {
            const auto found = namedMaterials.find(command.arguments.front());
            if (found != namedMaterials.end()) state.material = found->second;
        } else if (name == "AreaLightSource") {
            const std::string& type = command.arguments.front();
            if (!type.empty() && type != "diffuse")
                throw std::runtime_error("PBRT area light '" + type + "' is not supported by NoorRay");
            state.areaEmission = type.empty() ? glm::vec3(0.f)
                : rgb(command, "L", glm::vec3(1.f)) * scalar(command, "scale", 1.f);
        } else if (name == "ReverseOrientation") state.reverse = !state.reverse;
        else if (name == "Shape") {
                MaterialAuthoring material = state.material;
            if (glm::length2(state.areaEmission) > 0.f) {
                material.emission = state.areaEmission;
                material.emissionStrength = 1.f;
            }
            shapeTarget->push_back({command, state.transform, material, state.reverse});
        } else if (name == "ObjectBegin") {
            if (!currentObject.empty()) throw std::runtime_error("nested ObjectBegin is unsupported");
            currentObject = command.arguments.front();
            objectDefinitions[currentObject].clear();
            shapeTarget = &objectDefinitions[currentObject];
            stack.push_back(state);
        } else if (name == "ObjectEnd") {
            if (currentObject.empty()) throw std::runtime_error("ObjectEnd without ObjectBegin");
            currentObject.clear(); shapeTarget = &shapes;
            state = stack.back(); stack.pop_back();
        } else if (name == "ObjectInstance") {
            if (const auto found = objectDefinitions.find(command.arguments.front()); found != objectDefinitions.end())
                for (const ShapeRecord& definition : found->second) {
                    ShapeRecord instance = definition;
                    instance.transform = state.transform * definition.transform;
                    shapes.push_back(std::move(instance));
                }
        } else if (name == "LightSource") {
            const std::string& type = command.arguments.front();
            if (type != "infinite" && type != "point" && type != "spot" && type != "distant") {
                LOG_ERROR("PBRT light '" << type << "' is not supported; skipping "
                    << command.source.string() << ':' << command.line);
                continue;
            }
            const glm::vec3 color = rgb(command, "I", rgb(command, "L", glm::vec3(1.f)));
            const float intensity = scalar(command, "scale", 1.f);
            if (type == "infinite") {
                Environment& environment = scene.getEnvironment();
                const float colorMagnitude = std::max({color.r, color.g, color.b, 0.f});
                environment.color = colorMagnitude > 0.f
                    ? color / colorMagnitude : glm::vec3(0.f);
                // Despite the legacy member name, lightingExposure is a direct
                // multiplier. Keep color in display-friendly [0, 1] range and
                // preserve PBRT radiance as color * intensity.
                environment.lightingExposure = colorMagnitude * std::max(intensity, 0.f);
                const std::string filename = stringValue(command, "filename");
                if (!filename.empty()) {
                    const std::filesystem::path hdriPath =
                        (command.source.parent_path() / filename).lexically_normal();
                    if (!std::filesystem::exists(hdriPath)) {
                        LOG_WARN("PBRT HDRI not found; skipping: " << hdriPath.string()
                            << " (" << command.source.string() << ':' << command.line << ')');
                    } else {
                        scene.setEnvironmentTexture(
                            scene.add(Texture(hdriPath.string())));
                        environment.setEqualAreaMapping(glm::mat3(glm::inverse(state.transform)));
                    }
                }
                environment.updateDerivedSettings();
            } else {
                int lightType = type == "distant" ? LightInstance::TypeDirectional
                    : type == "spot" ? LightInstance::TypeSpot : LightInstance::TypePoint;
                glm::mat4 lightTransform = state.transform;
                if (type == "point" || type == "spot") {
                    const glm::vec3 from = transformedPoint(state.transform, point(command, "from", glm::vec3(0.f)));
                    lightTransform[3] = glm::vec4(from, 1.f);
                }
                if (type == "distant" || type == "spot") {
                    const glm::vec3 from = point(command, "from", glm::vec3(0.f));
                    const glm::vec3 to = point(command, "to", glm::vec3(0, 0, 1));
                    const glm::vec3 direction = glm::normalize(glm::mat3(state.transform) * (to - from));
                    lightTransform = glm::translate(glm::mat4(1.f), glm::vec3(lightTransform[3]))
                        * glm::toMat4(glm::rotation(glm::vec3(0, -1, 0), direction));
                }
                auto light = std::make_unique<LightInstance>(scene, type + "_light", Transform(lightTransform), lightType);
                light->setPhotometry(color, intensity);
                if (type == "point") light->setPointRadius(scalar(command, "radius", 0.f));
                if (type == "spot") {
                    light->setSpotRadius(scalar(command, "radius", 0.f));
                    const float outer = scalar(command, "coneangle", 30.f);
                    light->setSpotAngles(std::max(0.f, outer - scalar(command, "conedeltaangle", 5.f)), outer);
                }
                scene.add(std::move(light));
            }
        }
    }

    for (size_t i = 0; i < shapes.size(); ++i) addShape(scene, shapes[i], i);

    if (hasCamera) {
        std::unique_ptr<Camera> camera;
        if (cameraType == "orthographic")
            camera = std::make_unique<OrthographicCamera>();
        else if (cameraType == "perspective" && lensRadius > 0.f)
            camera = std::make_unique<ThinLensCamera>();
        else if (cameraType == "perspective")
            camera = std::make_unique<PerspectiveCamera>();
        else if (cameraType == "rossrealistic")
            camera = std::make_unique<RealisticCamera>();
        else if (cameraType == "rosspsf" || cameraType == "rosspsfcamera")
            camera = std::make_unique<HybridPsfCamera>();
        else
            throw std::runtime_error("PBRT camera '" + cameraType + "' is not supported by NoorRay");
        camera->setFocalLengthMm(camera->focalLengthMmForFovDegrees(cameraFov));
        camera->setFocusDistanceCm(focalDistance * 100.f);
        if (auto* thinLens = dynamic_cast<ThinLensCamera*>(camera.get()))
            thinLens->apertureDiameterMm = lensRadius * 2000.f;
        if ((dynamic_cast<RealisticCamera*>(camera.get())
                || dynamic_cast<HybridPsfCamera*>(camera.get())) && cameraCommand) {
            const Command& source = *cameraCommand;
            const std::string lens = relativeAssetPath(source, "lensfile");
            const std::string sensor = relativeAssetPath(source, "sensorFilePath");
            const std::string catalogs = relativeAssetList(source, "glasscatalogpaths");
            const bool isRealistic = dynamic_cast<RealisticCamera*>(camera.get()) != nullptr;
            if (lens.empty() || (!isRealistic && sensor.empty()))
                throw std::runtime_error(isRealistic
                    ? cameraType + " requires lensfile"
                    : cameraType + " requires lensfile and sensorFilePath");
            if (!sensor.empty())
                camera->getSensor().setImageSensorPath(sensor);
            else
                camera->getSensor().setDimensionsMm(
                    std::max(0.001f, scalar(source, "sensorwidthmm", camera->getSensor().width())),
                    std::max(0.001f, scalar(source, "sensorheightmm", camera->getSensor().height())));
            const OpticalSettings settings = opticalSettings(source, lens, sensor, catalogs);
            if (auto* realistic = dynamic_cast<RealisticCamera*>(camera.get())) {
                realistic->setOpticsPaths(lens, catalogs);
                realistic->setApertureDiameterMm(settings.apertureDiameterMm);
                realistic->setOpticalFocusDistanceCm(settings.focusDistanceCm);
                realistic->loadLensAndSensor(false);
            } else if (auto* hybridPsf = dynamic_cast<HybridPsfCamera*>(camera.get())) {
                hybridPsf->setOpticsPaths(lens, catalogs);
                hybridPsf->rayLutStepSize = std::max(1, static_cast<int>(scalar(source, "raylutstepsize", 32.f)));
                hybridPsf->samplesPerDimension = std::max(1, static_cast<int>(scalar(source, "samplesperdimension", 8.f)));
                hybridPsf->setApertureDiameterMm(settings.apertureDiameterMm);
                hybridPsf->setOpticalFocusDistanceCm(settings.focusDistanceCm);
                const std::string rayLut = relativeAssetPath(source, "raylut");
                hybridPsf->load(lens, catalogs, rayLut);
            }
        }
        // Loading a physical sensor restores its native resolution. PBRT's
        // Film resolution is the requested render extent and must win.
        camera->getSensor().setResolution(resolutionX, resolutionY);
        glm::mat4 worldFromCamera = glm::inverse(cameraFromWorld);
        // PBRT camera-space rays travel along +Z, while NoorRay camera-space
        // rays travel along -Z. Preserve PBRT's film X axis and only convert
        // the optical axis. In particular, PBRT scenes commonly use
        // `Scale -1 1 1` before Camera; flipping X again here would leave an
        // improper matrix that Transform's quaternion cannot represent.
        worldFromCamera[2] *= -1.f;
        auto cameraInstance = std::make_unique<CameraInstance>(std::move(camera), "PBRT Camera", Transform(worldFromCamera));
        scene.add(std::move(cameraInstance));
    }
}
