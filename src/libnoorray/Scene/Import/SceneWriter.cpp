#include "Scene/Import/SceneWriter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include <glaze/glaze.hpp>
#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>
#include <MaterialXFormat/XmlIo.h>

#include "Rendering/Camera/CameraInstance.h"
#include "Rendering/Camera/FisheyeCamera.h"
#include "Rendering/Camera/ThinLensCamera.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Scene/Objects/GaussianInstance.h"
#include "Scene/Objects/LightInstance.h"
#include "Scene/Objects/MeshInstance.h"
#include "Scene/Scene.h"
#include "Scene/Import/SceneFile.h"
#include "Scene/SceneObject.h"
#include "Scene/Import/SceneUsd.h"

namespace {
namespace mx = MaterialX;

nr::sceneio::Vec3 fromVec3(const glm::vec3 value)
{
    return {value.x, value.y, value.z};
}

std::string normalizedPath(const std::string& path)
{
    return std::filesystem::path(path).generic_string();
}

std::string lower(std::string value)
{
    std::ranges::transform(value, value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string primitiveTypeForMesh(const MeshAsset& asset)
{
    const std::string name = lower(asset.getName());
    if (name.find("cube") != std::string::npos) return "cube";
    if (name.find("plane") != std::string::npos) return "plane";
    if (name.find("disk") != std::string::npos) return "disk";
    if (name.find("sphere") != std::string::npos) return "sphere";
    return {};
}

nr::sceneio::EnvironmentFile makeEnvironmentFile(const Environment& environment)
{
    return {
        .color = fromVec3(environment.color),
        .lighting_exposure = environment.lightingExposure,
        .visible_exposure = environment.visibleExposure,
    };
}

nr::sceneio::RenderSettingsFile makeRenderSettingsFile(const RenderSettings& settings)
{
    return {
        .max_samples = settings.maxSamples,
        .aov_enabled = settings.aovEnabled,
        .optix_denoiser_enabled = settings.optixDenoiserEnabled,
        .optix_denoiser_min_samples = settings.optixDenoiserMinSamples,
        .gaussian_shading_mode = static_cast<int>(settings.gaussianShadingMode),
        .gaussian_render_sh_degree = static_cast<int>(settings.gaussianRenderSphericalHarmonics),
    };
}

nr::sceneio::CameraFile makeCameraFile(
    const CameraInstance& cameraInstance, const bool active)
{
    const Camera* camera = cameraInstance.getCamera();
    nr::sceneio::CameraFile file{};
    switch (cameraInstance.getProjectionType()) {
    case CameraProjectionType::Orthographic: file.projection = "orthographic"; break;
    case CameraProjectionType::Fisheye: file.projection = "fisheye"; break;
    case CameraProjectionType::ThinLens: file.projection = "thinlens"; break;
    case CameraProjectionType::Realistic: file.projection = "realistic"; break;
    case CameraProjectionType::HybridPsf: file.projection = "hybridpsf"; break;
    case CameraProjectionType::Perspective: file.projection = "perspective"; break;
    }
    file.active = active;
    if (!camera->Is<RealisticCamera>() && !camera->Is<HybridPsfCamera>())
        file.focal_length_mm = camera->getFocalLengthMm();
    file.focus_distance_cm = camera->getFocusDistanceCm();
    file.exposure = camera->exposure;
    if (const auto* thinLens = camera->CastOrNullptr<ThinLensCamera>()) {
        file.aperture_diameter_mm = thinLens->apertureDiameterMm;
        file.bokeh_bias = thinLens->bokehBias;
    } else if (const auto* fisheye = camera->CastOrNullptr<FisheyeCamera>()) {
        file.aperture_diameter_mm = fisheye->apertureDiameterMm;
        file.bokeh_bias = fisheye->bokehBias;
    } else if (const auto* realistic = camera->CastOrNullptr<RealisticCamera>()) {
        file.aperture_diameter_mm = realistic->apertureDiameterMm;
    } else if (const auto* hybridPsf = camera->CastOrNullptr<HybridPsfCamera>()) {
        file.aperture_diameter_mm = hybridPsf->apertureDiameterMm;
    }
    file.sensor_width_mm = camera->getSensor().width();
    file.sensor_height_mm = camera->getSensor().height();
    const glm::uvec2 resolution = camera->getSensor().resolution();
    file.resolution = {resolution.x, resolution.y};
    file.sensor = std::string(camera->getSensor().getImageSensorPath());
    switch (camera->getSensor().getType()) {
    case SensorType::ScatterPsf: file.sensor_type = "scatter_psf"; break;
    case SensorType::GatherPsf: file.sensor_type = "gather_psf"; break;
    case SensorType::Rectangular: file.sensor_type = "rectangular"; break;
    }
    file.psf = camera->getSensor().getPsfGridPath();
    if (const auto* realistic = camera->CastOrNullptr<RealisticCamera>()) {
        file.lens = realistic->getLensPath();
        file.glass_catalogs = realistic->getGlassCatalogPaths();
    } else if (const auto* hybridPsf = camera->CastOrNullptr<HybridPsfCamera>()) {
        file.lens = hybridPsf->getLensPath();
        file.glass_catalogs = hybridPsf->getGlassCatalogPaths();
        file.ray_lut = hybridPsf->getRayLutPath();
    }
    return file;
}

std::optional<nr::sceneio::ObjectFile> makeObjectFile(
    const std::shared_ptr<SceneObject>& object, const CameraInstance* activeCamera)
{
    if (!object)
        return std::nullopt;

    nr::sceneio::ObjectFile file{};
    file.type = object->getSourceType();
    file.name = object->getName();
    file.path = object->getSourcePath();

    if (const auto camera = std::dynamic_pointer_cast<CameraInstance>(object)) {
        file.type = "camera";
        file.camera = makeCameraFile(*camera, camera.get() == activeCamera);
    } else if (const auto gaussian = std::dynamic_pointer_cast<GaussianInstance>(object)) {
        if (file.path.empty())
            file.path = gaussian->getGaussianAsset().getPath();
        if (file.type.empty())
            file.type = "gaussian";
    } else if (const auto mesh = std::dynamic_pointer_cast<MeshInstance>(object)) {
        if (file.type.empty()) {
            file.type = primitiveTypeForMesh(mesh->getMeshAsset());
            if (file.type.empty())
                return std::nullopt;
        }
    } else if (file.type.empty()) {
        return std::nullopt;
    }

    if (!file.path.empty())
        file.path = normalizedPath(file.path);
    file.position = fromVec3(object->getPosition());
    file.rotation_euler = fromVec3(object->getRotationEuler());
    file.scale = fromVec3(object->getScale());
    return file;
}

nr::sceneio::SceneFile makeSceneFile(const Scene& scene)
{
    nr::sceneio::SceneFile file{};
    file.environment = makeEnvironmentFile(scene.getEnvironment());
    file.render_settings = makeRenderSettingsFile(scene.getRenderSettings());
    for (const auto& object : scene.getRootObjects())
        if (auto objectFile = makeObjectFile(object, scene.getActiveCamera()))
            file.objects.push_back(std::move(*objectFile));
    return file;
}

bool isPbrtFile(const std::string& filepath)
{
    std::string extension = std::filesystem::path(filepath).extension().string();
    std::ranges::transform(extension, extension.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension == ".pbrt";
}

std::string pbrtQuote(const std::string& value)
{
    std::string result;
    result.reserve(value.size() + 2);
    result.push_back('"');
    for (const char c : value) {
        if (c == '\\' || c == '"') result.push_back('\\');
        if (c == '\n') result += "\\n";
        else if (c == '\r') result += "\\r";
        else if (c == '\t') result += "\\t";
        else result.push_back(c);
    }
    result.push_back('"');
    return result;
}

std::string pbrtNumber(const float value)
{
    std::ostringstream stream;
    stream << std::setprecision(9) << value;
    return stream.str();
}

void writeVec3(std::ostream& out, const glm::vec3 value)
{
    out << pbrtNumber(value.x) << ' ' << pbrtNumber(value.y) << ' '
        << pbrtNumber(value.z);
}

void writeMatrix(std::ostream& out, const glm::mat4& matrix)
{
    out << "Transform [";
    // The PBRT reader follows PBRT's row-vector spelling and transposes those
    // rows into GLM columns. Writing matrix[row][column] therefore reproduces
    // the original GLM matrix on import.
    for (int row = 0; row < 4; ++row) {
        if (row != 0) out << ' ';
        for (int column = 0; column < 4; ++column) {
            if (column != 0) out << ' ';
            out << pbrtNumber(matrix[row][column]);
        }
    }
    out << "]\n";
}

std::vector<float> materialNumbers(std::string value)
{
    for (char& c : value)
        if (c == ',') c = ' ';
    std::istringstream stream(value);
    std::vector<float> result;
    for (float number; stream >> number;)
        result.push_back(number);
    return result;
}

const mx::InputPtr findMaterialInput(const mx::NodePtr& shader,
    const std::initializer_list<std::string_view> names)
{
    if (!shader) return {};
    for (const std::string_view name : names)
        if (const mx::InputPtr input = shader->getInput(std::string(name)))
            return input;
    return {};
}

float materialFloat(const mx::NodePtr& shader,
    const std::initializer_list<std::string_view> names, const float fallback)
{
    const mx::InputPtr input = findMaterialInput(shader, names);
    if (!input || !input->getValue()) return fallback;
    const std::vector<float> values = materialNumbers(input->getValueString());
    return values.empty() ? fallback : values.front();
}

glm::vec3 materialColor(const mx::NodePtr& shader,
    const std::initializer_list<std::string_view> names, const glm::vec3 fallback)
{
    const mx::InputPtr input = findMaterialInput(shader, names);
    if (!input || !input->getValue()) return fallback;
    const std::vector<float> values = materialNumbers(input->getValueString());
    if (values.size() == 1) return glm::vec3(values.front());
    return values.size() >= 3 ? glm::vec3(values[0], values[1], values[2]) : fallback;
}

struct PbrtMaterial {
    std::string type{"diffuse"};
    glm::vec3 reflectance{0.8f};
    float roughness{0.5f};
    float eta{1.5f};
    glm::vec3 emission{};
    float emissionStrength{};
};

mx::DocumentPtr loadMaterialDocument(const Scene& scene, const uint32_t slot,
    const std::filesystem::path& outputPath)
{
    const auto& documents = scene.getMaterialXDocuments();
    if (slot < documents.size() && documents[slot]) return documents[slot];

    const auto& paths = scene.getMaterialXSourcePaths();
    if (slot >= paths.size() || paths[slot].empty()) return {};
    std::filesystem::path source(paths[slot]);
    if (source.is_relative()) source = std::filesystem::absolute(source);
    if (!std::filesystem::is_regular_file(source)) return {};
    const mx::DocumentPtr document = mx::createDocument();
    mx::readFromXmlString(document, [&] {
        std::ifstream input(source, std::ios::binary);
        std::ostringstream contents;
        contents << input.rdbuf();
        return contents.str();
    }());
    (void)outputPath;
    return document;
}

PbrtMaterial pbrtMaterial(const Scene& scene, const uint32_t slot,
    const std::filesystem::path& outputPath)
{
    PbrtMaterial result;
    const mx::DocumentPtr document = loadMaterialDocument(scene, slot, outputPath);
    if (!document) return result;

    mx::NodePtr shader;
    for (const mx::NodePtr& node : document->getNodes()) {
        if (node->getType() != "surfaceshader") continue;
        if (!shader || node->getCategory() == "open_pbr_surface"
            || node->getCategory() == "standard_surface"
            || node->getCategory() == "disney_principled")
            shader = node;
    }
    if (!shader) return result;

    const std::string category = shader->getCategory();
    if (category == "open_pbr_surface" || category == "standard_surface") {
        result.reflectance = materialColor(shader,
            {"base_color", "baseColor", "basecolor"}, result.reflectance);
        result.roughness = std::clamp(materialFloat(shader,
            {"specular_roughness", "roughness"}, result.roughness), 0.f, 1.f);
        const float metallic = std::clamp(materialFloat(shader,
            {"base_metalness", "metalness", "metallic"}, 0.f), 0.f, 1.f);
        const float transmission = std::clamp(materialFloat(shader,
            {"transmission_weight", "transmission"}, 0.f), 0.f, 1.f);
        if (transmission > 0.5f) {
            result.type = "dielectric";
            result.eta = std::max(1.f, materialFloat(shader, {"ior"}, 1.5f));
        } else if (metallic > 0.5f) {
            result.type = "conductor";
        } else {
            result.type = "coateddiffuse";
        }
        result.emission = materialColor(shader,
            {"emission_color", "emission"}, result.emission);
        result.emissionStrength = materialFloat(shader,
            {"emission_luminance", "emission_strength"}, 0.f);
    } else if (category == "disney_principled") {
        result.type = "coateddiffuse";
        result.reflectance = materialColor(shader,
            {"baseColor", "base_color"}, result.reflectance);
        result.roughness = std::clamp(materialFloat(shader,
            {"roughness"}, result.roughness), 0.f, 1.f);
    }
    return result;
}

void writePbrtMaterial(std::ostream& out, const PbrtMaterial& material)
{
    out << "Material " << pbrtQuote(material.type);
    if (material.type == "dielectric") {
        out << " \"float eta\" [" << pbrtNumber(material.eta) << "]";
    } else {
        out << " \"rgb reflectance\" [";
        writeVec3(out, material.reflectance);
        out << "] \"float roughness\" [" << pbrtNumber(material.roughness) << "]";
    }
    out << "\n";
}

void writePbrtShape(std::ostream& out, const MeshAsset& asset,
    const std::vector<uint32_t>& indices)
{
    out << "Shape \"trianglemesh\" \"integer indices\" [";
    for (size_t i = 0; i < indices.size(); ++i)
        out << (i ? " " : "") << indices[i];
    out << "] \"point3 P\" [";
    for (size_t i = 0; i < asset.getVertices().size(); ++i) {
        if (i) out << ' ';
        writeVec3(out, asset.getVertices()[i].position);
    }
    out << "] \"normal N\" [";
    for (size_t i = 0; i < asset.getVertices().size(); ++i) {
        if (i) out << ' ';
        writeVec3(out, asset.getVertices()[i].normal);
    }
    out << "] \"point2 uv\" [";
    for (size_t i = 0; i < asset.getVertices().size(); ++i) {
        if (i) out << ' ';
        const glm::vec2 uv = asset.getVertices()[i].uv;
        out << pbrtNumber(uv.x) << ' ' << pbrtNumber(1.f - uv.y);
    }
    out << "]\n";
}

void writePbrtMesh(std::ostream& out, const Scene& scene,
    const MeshInstance& instance, const std::filesystem::path& outputPath)
{
    if (!instance.isVisible() || !instance.hasMeshAsset()) return;
    const MeshAsset& asset = instance.getMeshAsset();
    const size_t materialCount = std::max<size_t>(asset.getMaterialCount(), 1);
    std::vector<std::vector<uint32_t>> indicesByMaterial(materialCount);
    const auto& indices = asset.getIndices();
    const auto& faces = asset.getFaces();
    for (size_t triangle = 0; triangle * 3 + 2 < indices.size(); ++triangle) {
        const size_t indexOffset = triangle * 3;
        const uint32_t material = triangle < faces.size()
            ? std::min<uint32_t>(faces[triangle].materialIndex,
                static_cast<uint32_t>(materialCount - 1)) : 0;
        auto& target = indicesByMaterial[material];
        target.insert(target.end(), {indices[indexOffset], indices[indexOffset + 1], indices[indexOffset + 2]});
    }

    for (size_t material = 0; material < indicesByMaterial.size(); ++material) {
        if (indicesByMaterial[material].empty()) continue;
        out << "AttributeBegin\n";
        writeMatrix(out, instance.getWorldTransform().getMatrix());
        const uint32_t slot = material < asset.getMaterialCount()
            ? asset.getMaterialIds()[material] : 0;
        const PbrtMaterial pbrt = pbrtMaterial(scene, slot, outputPath);
        writePbrtMaterial(out, pbrt);
        if (pbrt.emissionStrength > 0.f) {
            out << "AreaLightSource \"diffuse\" \"rgb L\" [";
            writeVec3(out, pbrt.emission);
            out << "] \"float scale\" [" << pbrtNumber(pbrt.emissionStrength) << "]\n";
        }
        writePbrtShape(out, asset, indicesByMaterial[material]);
        out << "AttributeEnd\n";
    }
}

void writePbrtObject(std::ostream& out, const Scene& scene,
    const std::shared_ptr<SceneObject>& object, const std::filesystem::path& outputPath)
{
    if (!object) return;
    if (const auto mesh = std::dynamic_pointer_cast<MeshInstance>(object))
        writePbrtMesh(out, scene, *mesh, outputPath);
    else if (std::dynamic_pointer_cast<GaussianInstance>(object))
        out << "# NoorRay: Gaussian splat objects are not representable in PBRT and were skipped: "
            << pbrtQuote(object->getName()) << "\n";
    for (const auto& child : object->getChildren())
        writePbrtObject(out, scene, child, outputPath);
}

void writePbrtLight(std::ostream& out, const LightInstance& instance)
{
    const auto& data = instance.getLightData();
    if (instance.lightType == LightInstance::TypeRect) {
        const auto* light = std::get_if<RectLight>(&data);
        if (!light) return;
        const glm::vec3 normal = glm::normalize(light->direction);
        const glm::vec3 tangent = glm::normalize(light->tangent);
        const glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangent));
        glm::mat4 transform(
            glm::vec4(tangent * light->width, 0.f),
            glm::vec4(bitangent * light->height, 0.f),
            glm::vec4(normal, 0.f),
            glm::vec4(light->position, 1.f));
        out << "AttributeBegin\n";
        writeMatrix(out, transform);
        out << "AreaLightSource \"diffuse\" \"rgb L\" [";
        writeVec3(out, light->color);
        out << "] \"float scale\" [" << pbrtNumber(light->intensity) << "]\n";
        out << "Shape \"trianglemesh\" \"integer indices\" [0 1 2 0 2 3] "
            << "\"point3 P\" [-0.5 -0.5 0 0.5 -0.5 0 0.5 0.5 0 -0.5 0.5 0]\n";
        out << "AttributeEnd\n";
        return;
    }

    const glm::vec3 color = instance.getColor();
    if (const auto* point = std::get_if<PointLight>(&data)) {
        out << "LightSource \"point\" \"point from\" [";
        writeVec3(out, point->position);
        out << "] \"rgb I\" [";
        writeVec3(out, color);
        out << "] \"float scale\" [" << pbrtNumber(point->intensity)
            << "] \"float radius\" [" << pbrtNumber(point->softRadius) << "]\n";
    } else if (const auto* spot = std::get_if<SpotLight>(&data)) {
        out << "LightSource \"spot\" \"point from\" [";
        writeVec3(out, spot->position);
        out << "] \"point to\" [";
        writeVec3(out, spot->position + glm::normalize(spot->direction));
        out << "] \"rgb I\" [";
        writeVec3(out, color);
        out << "] \"float scale\" [" << pbrtNumber(spot->intensity)
            << "] \"float radius\" [" << pbrtNumber(spot->softRadius)
            << "] \"float coneangle\" [" << pbrtNumber(spot->outerConeAngle)
            << "] \"float conedeltaangle\" ["
            << pbrtNumber(spot->outerConeAngle - spot->innerConeAngle) << "]\n";
    } else if (const auto* directional = std::get_if<DirectionalLight>(&data)) {
        out << "LightSource \"distant\" \"point from\" [0 0 0] \"point to\" [";
        writeVec3(out, directional->direction);
        out << "] \"rgb L\" [";
        writeVec3(out, color);
        out << "] \"float scale\" [" << pbrtNumber(directional->intensity) << "]\n";
    }
}

void writePbrtLights(std::ostream& out, const std::shared_ptr<SceneObject>& object)
{
    if (!object) return;
    if (const auto light = std::dynamic_pointer_cast<LightInstance>(object))
        if (light->isVisible()) writePbrtLight(out, *light);
    for (const auto& child : object->getChildren())
        writePbrtLights(out, child);
}

void writePbrtCamera(std::ostream& out, const CameraInstance& instance,
    const int maxSamples, const std::filesystem::path& outputPath)
{
    const Camera& camera = *instance.getCamera();
    const glm::vec3 position = instance.getPosition();
    const glm::vec3 direction = glm::normalize(instance.getRotation() * CameraInstance::LocalForward);
    const glm::vec3 up = glm::normalize(instance.getRotation() * CameraInstance::LocalUp);
    out << "LookAt ";
    writeVec3(out, position);
    out << ' ';
    // PBRT camera rays travel along +Z, whereas NoorRay camera rays travel
    // along -Z. The importer applies the matching Z flip after LookAt.
    writeVec3(out, position - direction);
    out << ' ';
    writeVec3(out, up);
    out << "\n";

    const auto resolution = camera.getSensor().resolution();
    out << "Film \"rgb\" \"integer xresolution\" [" << resolution.x
        << "] \"integer yresolution\" [" << resolution.y << "]\n";
    out << "Sampler \"independent\" \"integer pixelsamples\" ["
        << std::max(maxSamples, 1) << "]\n";
    const float fov = camera.fovDegreesForFocalLengthMm(camera.getFocalLengthMm());
    if (instance.getProjectionType() == CameraProjectionType::Orthographic) {
        out << "Camera \"orthographic\" \"float fov\" [" << pbrtNumber(fov) << "]\n";
    } else if (instance.getProjectionType() == CameraProjectionType::ThinLens) {
        const auto* thinLens = camera.CastOrNullptr<ThinLensCamera>();
        out << "Camera \"perspective\" \"float fov\" [" << pbrtNumber(fov)
            << "] \"float lensradius\" ["
            << pbrtNumber(thinLens ? thinLens->apertureDiameterMm / 2000.f : 0.f)
            << "] \"float focaldistance\" ["
            << pbrtNumber(camera.getFocusDistanceCm() / 100.f) << "]\n";
    } else if (instance.getProjectionType() == CameraProjectionType::Realistic) {
        const auto* realistic = camera.CastOrNullptr<RealisticCamera>();
        if (!realistic || realistic->getLensPath().empty()) {
            out << "# NoorRay: realistic camera has no lens path; exported as perspective\n";
            out << "Camera \"perspective\" \"float fov\" [" << pbrtNumber(fov) << "]\n";
        } else {
            out << "Camera \"rossrealistic\" \"string lensfile\" ["
                << pbrtQuote(std::filesystem::path(realistic->getLensPath()).generic_string())
                << "] \"float aperturediameter\" ["
                << pbrtNumber(realistic->apertureDiameterMm) << "] \"float focusdistance\" ["
                << pbrtNumber(camera.getFocusDistanceCm() / 100.f) << "]\n";
        }
    } else if (instance.getProjectionType() == CameraProjectionType::HybridPsf) {
        const auto* hybrid = camera.CastOrNullptr<HybridPsfCamera>();
        if (!hybrid || hybrid->getLensPath().empty() || hybrid->getSensor().getImageSensorPath().empty()) {
            out << "# NoorRay: hybrid PSF camera is missing optical assets; exported as perspective\n";
            out << "Camera \"perspective\" \"float fov\" [" << pbrtNumber(fov) << "]\n";
        } else {
            out << "Camera \"rosspsf\" \"string lensfile\" ["
                << pbrtQuote(std::filesystem::path(hybrid->getLensPath()).generic_string())
                << "] \"string sensorFilePath\" ["
                << pbrtQuote(std::filesystem::path(hybrid->getSensor().getImageSensorPath()).generic_string())
                << "] \"float aperturediameter\" ["
                << pbrtNumber(hybrid->apertureDiameterMm) << "] \"float focusdistance\" ["
                << pbrtNumber(camera.getFocusDistanceCm() / 100.f) << "]\n";
        }
    } else {
        out << "Camera \"perspective\" \"float fov\" [" << pbrtNumber(fov) << "]\n";
    }
    (void)outputPath;
}

void writePbrt(const Scene& scene, const std::string& filepath)
{
    std::ofstream out(filepath, std::ios::binary);
    if (!out)
        throw std::runtime_error("Failed to open PBRT scene for writing: " + filepath);
    const std::filesystem::path outputPath(filepath);
    out << "# NoorRay PBRT scene export\n"
        << "Integrator \"path\"\n"
        << "PixelFilter \"box\"\n";
    writePbrtCamera(out, *scene.getRenderCamera(), scene.getRenderSettings().maxSamples, outputPath);
    out << "WorldBegin\n";
    const Environment& environment = scene.getEnvironment();
    out << "LightSource \"infinite\" \"rgb L\" [";
    writeVec3(out, environment.color);
    out << "] \"float scale\" [" << pbrtNumber(environment.lightingExposure) << "]\n";
    for (const auto& object : scene.getRootObjects()) {
        writePbrtObject(out, scene, object, outputPath);
        writePbrtLights(out, object);
    }
    out << "WorldEnd\n";
    if (!out)
        throw std::runtime_error("Failed to write PBRT scene: " + filepath);
}
}

void SceneWriter::Write(const Scene& scene, const std::string& filepath)
{
    if (nr::sceneio::isUsdFile(filepath)) {
        nr::sceneio::writeUsd(scene, filepath);
        return;
    }
    if (isPbrtFile(filepath)) {
        writePbrt(scene, filepath);
        return;
    }
    const nr::sceneio::SceneFile sceneFile = makeSceneFile(scene);
    std::string json;
    constexpr glz::opts writeOptions{.prettify = true};
    if (const auto error = glz::write<writeOptions>(sceneFile, json))
        throw std::runtime_error("Failed to serialize scene JSON: " + glz::format_error(error));

    std::ofstream out(filepath, std::ios::binary);
    if (!out)
        throw std::runtime_error("Failed to open scene file for writing: " + filepath);
    out << json << '\n';
    if (!out)
        throw std::runtime_error("Failed to write scene file: " + filepath);
}
