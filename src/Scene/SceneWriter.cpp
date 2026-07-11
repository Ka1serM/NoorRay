#include "Scene/SceneWriter.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <glaze/glaze.hpp>

#include "Camera/CameraInstance.h"
#include "Camera/FisheyeCamera.h"
#include "Camera/ThinLensCamera.h"
#include "Mesh/MeshAsset.h"
#include "Scene/GaussianInstance.h"
#include "Scene/MeshInstance.h"
#include "Scene/Scene.h"
#include "Scene/SceneFile.h"
#include "Scene/SceneObject.h"

namespace {
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
        .visible = environment.visible != 0,
    };
}

nr::sceneio::RenderSettingsFile makeRenderSettingsFile(const RenderSettings& settings)
{
    return {
        .max_samples = settings.maxSamples,
        .gaussian_shading_mode = static_cast<int>(settings.gaussianShadingMode),
    };
}

nr::sceneio::CameraFile makeCameraFile(const CameraInstance& cameraInstance)
{
    const Camera* camera = cameraInstance.getCamera();
    nr::sceneio::CameraFile file{};
    switch (cameraInstance.getProjectionType()) {
    case CameraProjectionType::Orthographic: file.type = "orthographic"; break;
    case CameraProjectionType::Fisheye: file.type = "fisheye"; break;
    case CameraProjectionType::ThinLens: file.type = "thinlens"; break;
    case CameraProjectionType::Realistic: file.type = "realistic"; break;
    case CameraProjectionType::RossPsf: file.type = "rosspsf"; break;
    case CameraProjectionType::Perspective: file.type = "perspective"; break;
    }

    file.position = fromVec3(cameraInstance.getPosition());
    file.rotation_euler = fromVec3(cameraInstance.getRotationEuler());
    file.scale = fromVec3(cameraInstance.getScale());
    file.focal_length = camera->focalLengthMm;
    file.focus_distance = camera->focusDistance;
    if (const auto* thinLens = camera->CastOrNullptr<ThinLensCamera>()) {
        file.aperture_diameter = thinLens->fStop;
        file.bokeh_bias = thinLens->bokehBias;
    } else if (const auto* fisheye = camera->CastOrNullptr<FisheyeCamera>()) {
        file.bokeh_bias = fisheye->bokehBias;
    }
    const glm::uvec2 resolution = camera->getSensor().resolution();
    file.resolution = {resolution.x, resolution.y};
    file.sensor = std::string(camera->getSensor().getImageSensorPath());
    if (const auto* realistic = camera->CastOrNullptr<RealisticCamera>()) {
        file.lens = realistic->getLensPath();
        file.glass_catalogs = realistic->getGlassCatalogPaths();
    } else if (const auto* rossPsf = camera->CastOrNullptr<RossPsfCamera>()) {
        file.lens = rossPsf->getLensPath();
        file.glass_catalogs = rossPsf->getGlassCatalogPaths();
    }
    return file;
}

std::optional<nr::sceneio::ObjectFile> makeObjectFile(const std::shared_ptr<SceneObject>& object)
{
    if (!object || dynamic_cast<CameraInstance*>(object.get()))
        return std::nullopt;

    nr::sceneio::ObjectFile file{};
    file.type = object->getSourceType();
    file.name = object->getName();
    file.path = object->getSourcePath();

    if (const auto gaussian = std::dynamic_pointer_cast<GaussianInstance>(object)) {
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
    if (const CameraInstance* camera = scene.getActiveCamera())
        file.camera = makeCameraFile(*camera);
    for (const auto& object : scene.getRootObjects())
        if (auto objectFile = makeObjectFile(object))
            file.objects.push_back(std::move(*objectFile));
    return file;
}
}

void SceneWriter::Write(const Scene& scene, const std::string& filepath)
{
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
