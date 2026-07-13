#include "Scene/SceneReader.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <glaze/glaze.hpp>

#include "Camera/CameraInstance.h"
#include "Camera/FisheyeCamera.h"
#include "Camera/OrthographicCamera.h"
#include "Camera/PerspectiveCamera.h"
#include "Camera/RealisticCamera.h"
#include "Camera/RossPsfCamera.h"
#include "Camera/ThinLensCamera.h"
#include "Mesh/MeshAsset.h"
#include "Mesh/Transform.h"
#include "Raytracing/Sellmeier.h"
#include "Scene/MeshInstance.h"
#include "Scene/Scene.h"
#include "Scene/SceneFile.h"
#include "Scene/SceneImporter.h"
#include "Scene/SceneObject.h"

namespace {
glm::vec3 toVec3(const nr::sceneio::Vec3& value)
{
    return {value[0], value[1], value[2]};
}

Transform toTransform(const nr::sceneio::ObjectFile& object)
{
    return {toVec3(object.position), toVec3(object.rotation_euler), toVec3(object.scale)};
}

std::vector<std::string> splitPaths(const std::string& paths)
{
    std::vector<std::string> result;
    std::stringstream stream(paths);
    for (std::string path; std::getline(stream, path, ';');)
        if (!path.empty()) result.push_back(std::move(path));
    return result;
}

SensorType sensorType(const std::string& type)
{
    if (type == "scatter_psf") return SensorType::ScatterPsf;
    if (type == "gather_psf") return SensorType::GatherPsf;
    return SensorType::Rectangular;
}

Material toMaterial(const nr::sceneio::MaterialFile& file)
{
    Material material{};
    material.albedo = toVec3(file.albedo);
    material.roughness = file.roughness;
    material.metallic = file.metallic;
    material.specular = file.specular;
    const glm::vec3 iorRgb(file.ior_r, file.ior_g, file.ior_b);
    material.sellmeier = fitSellmeierFromFraunhofer(iorRgb);
    material.transmission = file.transmission;
    material.opacity = file.opacity;
    material.emission = toVec3(file.emission);
    material.emissionStrength = file.emission_strength;
    return material;
}

std::unique_ptr<Camera> makeCamera(const nr::sceneio::CameraFile& file)
{
    CameraProjectionType projection = CameraProjectionType::Perspective;
    if (file.type == "realistic") projection = CameraProjectionType::Realistic;
    else if (file.type == "rosspsf" || file.type == "hybridpsf")
        projection = CameraProjectionType::HybridPsf;
    else if (file.type == "thinlens") projection = CameraProjectionType::ThinLens;
    else if (file.type == "fisheye") projection = CameraProjectionType::Fisheye;
    else if (file.type == "orthographic") projection = CameraProjectionType::Orthographic;

    std::unique_ptr<Sensor> sensor;
    const SensorType type = sensorType(file.sensor_type);
    if (type == SensorType::ScatterPsf)
        sensor = std::make_unique<ScatterPsfSensor>();
    else if (type == SensorType::GatherPsf)
        sensor = std::make_unique<GatherPsfSensor>();
    else
        sensor = std::make_unique<RectangularSensor>();
    std::unique_ptr<Camera> camera = Camera::create(projection, std::move(sensor));
    if (auto* realistic = dynamic_cast<RealisticCamera*>(camera.get()))
        realistic->apertureDiameterMm = file.aperture_diameter;
    else if (auto* hybrid = dynamic_cast<RossPsfCamera*>(camera.get()))
        hybrid->apertureDiameterMm = file.aperture_diameter;
    else if (auto* thinLens = dynamic_cast<ThinLensCamera*>(camera.get())) {
        thinLens->fStop = file.aperture_diameter;
        thinLens->bokehBias = std::max(0.001f, file.bokeh_bias);
    } else if (auto* fisheye = dynamic_cast<FisheyeCamera*>(camera.get())) {
        fisheye->fStop = file.aperture_diameter;
        fisheye->bokehBias = std::max(0.001f, file.bokeh_bias);
    }

    camera->setFocalLength(file.focal_length);
    camera->setFocusDistance(file.focus_distance);
    camera->getSensor().setImageSensorPath(file.sensor);
    camera->getSensor().setResolution(
        std::max(file.resolution[0], 1u),
        std::max(file.resolution[1], 1u));
    return camera;
}

void addCamera(Scene& scene, const nr::sceneio::CameraFile& file)
{
    std::unique_ptr<Camera> camera = makeCamera(file);
    if (file.type == "realistic" && !file.lens.empty() && !file.sensor.empty())
        dynamic_cast<RealisticCamera&>(*camera).load(
            file.lens, splitPaths(file.glass_catalogs));
    else if ((file.type == "rosspsf" || file.type == "hybridpsf")
        && !file.lens.empty() && !file.sensor.empty()) {
        Sensor& sensor = camera->getSensor();
        if (!file.psf.empty())
            sensor.setPsfGridPath(file.psf);
        dynamic_cast<RossPsfCamera&>(*camera).load(
            file.lens, splitPaths(file.glass_catalogs), file.ray_lut);
    }
    Transform transform{toVec3(file.position), toVec3(file.rotation_euler), toVec3(file.scale)};
    auto cameraInstance = std::make_unique<CameraInstance>(
        std::move(camera), "Camera", transform);
    scene.add(std::move(cameraInstance));
}

void addObject(Scene& scene, const nr::sceneio::ObjectFile& object)
{
    const Transform transform = toTransform(object);
    if (object.type == "gltf" || object.type == "glb") {
        SceneImporter::ImportGltfScene(scene, object.path);
        if (SceneObject* root = scene.getActiveObject())
            root->setLocalTransform(transform);
    } else if (object.type == "ply" || object.type == "gaussian" || object.type == "3dgs" ||
               object.type == "splat" || object.type == "ksplat" || object.type == "spz" || object.type == "sog") {
        SceneImporter::ImportGaussianScene(scene, object.path);
        if (SceneObject* root = scene.getActiveObject())
            root->setLocalTransform(transform);
    } else if (object.type == "obj") {
        Material materialOverride{};
        const Material* materialOverridePtr = nullptr;
        if (object.material) {
            materialOverride = toMaterial(*object.material);
            materialOverridePtr = &materialOverride;
        }
        SceneImporter::ImportObjScene(scene, object.path, materialOverridePtr);
        if (SceneObject* root = scene.getActiveObject())
            root->setLocalTransform(transform);
    } else {
        Material material = object.material ? toMaterial(*object.material) : Material{};
        uint32_t meshIndex;
        if (object.type == "cube")
            meshIndex = scene.add(MeshAsset::CreateCube(scene, object.name, material));
        else if (object.type == "plane")
            meshIndex = scene.add(MeshAsset::CreatePlane(scene, object.name, material));
        else if (object.type == "disk")
            meshIndex = scene.add(MeshAsset::CreateDisk(scene, object.name, material));
        else
            meshIndex = scene.add(MeshAsset::CreateSphere(scene, object.name, material));
        scene.add(std::make_unique<MeshInstance>(scene, object.name, meshIndex, transform));
    }
}

std::string readTextFile(const std::string& filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open scene file: " + filepath);
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}
}

void SceneReader::Read(Scene& scene, const std::string& filepath)
{
    nr::sceneio::SceneFile file{};
    std::string json = readTextFile(filepath);
    constexpr glz::opts readOptions{.error_on_unknown_keys = false};
    if (const auto error = glz::read<readOptions>(file, json))
        throw std::runtime_error("Failed to parse scene JSON: " + glz::format_error(error, json));

    scene.clear();

    if (file.environment) {
        Environment& environment = scene.getEnvironment();
        environment.color = toVec3(file.environment->color);
        environment.lightingExposure = file.environment->lighting_exposure;
        environment.visibleExposure = file.environment->visible_exposure;
        environment.visible = file.environment->visible ? 1 : 0;
        environment.updateDerivedSettings();
    }

    if (file.render_settings) {
        scene.getRenderSettings().maxSamples = file.render_settings->max_samples;
        scene.getRenderSettings().noiseLimitEnabled = file.render_settings->noise_limit_enabled;
        scene.getRenderSettings().noiseLevel = std::max(file.render_settings->noise_level, 0.0f);
        scene.getRenderSettings().gaussianShadingMode =
            file.render_settings->gaussian_shading_mode == static_cast<int>(GaussianShadingMode::DirectColor)
                ? GaussianShadingMode::DirectColor
                : GaussianShadingMode::GlobalIllumination;
        scene.getRenderSettings().gaussianRenderSphericalHarmonics =
            clampSphericalHarmonicsOrder(file.render_settings->gaussian_render_sh_degree);
    }

    if (file.camera)
        addCamera(scene, *file.camera);

    for (const nr::sceneio::ObjectFile& object : file.objects)
        addObject(scene, object);
}
