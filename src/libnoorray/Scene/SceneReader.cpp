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
#include "Camera/HybridPsfCamera.h"
#include "Camera/ThinLensCamera.h"
#include "Mesh/Assets/MeshAsset.h"
#include "Mesh/Transform.h"
#include "MaterialX/MaterialXCompiler.h"
#include "Shading/Sellmeier.h"
#include "Scene/MeshInstance.h"
#include "Scene/Scene.h"
#include "Scene/SceneFile.h"
#include "Scene/SceneImporter.h"
#include "Scene/SceneObject.h"

namespace {
std::string readTextFile(const std::string& filepath);
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
    if (type == "rectangular") return SensorType::Rectangular;
    throw std::runtime_error("Unknown camera sensor type: " + type);
}

MaterialAuthoring toMaterial(const nr::sceneio::MaterialFile& file)
{
    MaterialAuthoring material{};
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
    std::unique_ptr<Sensor> sensor;
    const SensorType type = sensorType(file.sensor_type);
    if (type == SensorType::ScatterPsf)
        sensor = std::make_unique<ScatterPsfSensor>();
    else if (type == SensorType::GatherPsf)
        sensor = std::make_unique<GatherPsfSensor>();
    else
        sensor = std::make_unique<RectangularSensor>();
    std::unique_ptr<Camera> camera;
    if (file.projection == "realistic")
        camera = std::make_unique<RealisticCamera>(std::move(sensor));
    else if (file.projection == "hybridpsf")
        camera = std::make_unique<HybridPsfCamera>(std::move(sensor));
    else if (file.projection == "thinlens")
        camera = std::make_unique<ThinLensCamera>(std::move(sensor));
    else if (file.projection == "fisheye")
        camera = std::make_unique<FisheyeCamera>(std::move(sensor));
    else if (file.projection == "orthographic")
        camera = std::make_unique<OrthographicCamera>(std::move(sensor));
    else if (file.projection == "perspective")
        camera = std::make_unique<PerspectiveCamera>(std::move(sensor));
    else
        throw std::runtime_error("Unknown camera projection: " + file.projection);
    if (auto* realistic = dynamic_cast<RealisticCamera*>(camera.get()))
        realistic->apertureDiameterMm = file.aperture_diameter_mm;
    else if (auto* hybrid = dynamic_cast<HybridPsfCamera*>(camera.get()))
        hybrid->apertureDiameterMm = file.aperture_diameter_mm;
    else if (auto* thinLens = dynamic_cast<ThinLensCamera*>(camera.get())) {
        thinLens->apertureDiameterMm = file.aperture_diameter_mm;
        thinLens->bokehBias = std::max(0.001f, file.bokeh_bias);
    } else if (auto* fisheye = dynamic_cast<FisheyeCamera*>(camera.get())) {
        fisheye->apertureDiameterMm = file.aperture_diameter_mm;
        fisheye->bokehBias = std::max(0.001f, file.bokeh_bias);
    }

    camera->setFocalLengthMm(file.focal_length_mm);
    camera->setFocusDistanceCm(file.focus_distance_cm);
    camera->exposure = file.exposure;
    camera->getSensor().setImageSensorPath(file.sensor);
    camera->getSensor().setDimensionsMm(
        std::max(file.sensor_width_mm, 0.001f),
        std::max(file.sensor_height_mm, 0.001f));
    camera->getSensor().setResolution(
        std::max(file.resolution[0], 1u),
        std::max(file.resolution[1], 1u));
    return camera;
}

CameraInstance* addCamera(Scene& scene, const nr::sceneio::ObjectFile& object)
{
    if (!object.camera)
        throw std::runtime_error("Camera object is missing its camera properties");
    const nr::sceneio::CameraFile& file = *object.camera;
    std::unique_ptr<Camera> camera = makeCamera(file);
    if (file.projection == "realistic") {
        if (file.lens.empty())
            throw std::runtime_error("Realistic camera requires a lens path");
        dynamic_cast<RealisticCamera&>(*camera).load(
            file.lens, splitPaths(file.glass_catalogs));
    } else if (file.projection == "hybridpsf") {
        if (file.lens.empty() || file.sensor.empty())
            throw std::runtime_error("Hybrid PSF camera requires lens and sensor paths");
        Sensor& sensor = camera->getSensor();
        if (!file.psf.empty())
            sensor.setPsfGridPath(file.psf);
        dynamic_cast<HybridPsfCamera&>(*camera).load(
            file.lens, splitPaths(file.glass_catalogs), file.ray_lut);
    }
    const Transform transform = toTransform(object);
    auto cameraInstance = std::make_unique<CameraInstance>(
        std::move(camera), object.name, transform);
    CameraInstance* result = cameraInstance.get();
    scene.add(std::move(cameraInstance));
    if (file.active)
        scene.setActiveCamera(result);
    return result;
}

void addObject(Scene& scene, const nr::sceneio::ObjectFile& object)
{
    const Transform transform = toTransform(object);
    if (object.type == "camera") {
        addCamera(scene, object);
    } else if (object.camera) {
        throw std::runtime_error("Only camera objects may contain camera properties");
    } else if (object.type == "gltf" || object.type == "glb") {
        SceneImporter::ImportGltfScene(scene, object.path);
        if (SceneObject* root = scene.getActiveObject())
            root->setLocalTransform(transform);
    } else if (object.type == "ply" || object.type == "splat" || object.type == "ksplat"
        || object.type == "spz" || object.type == "sog") {
        SceneImporter::ImportGaussianScene(scene, object.path);
        if (SceneObject* root = scene.getActiveObject())
            root->setLocalTransform(transform);
    } else if (object.type == "obj") {
        MaterialAuthoring materialOverride{};
        const MaterialAuthoring* materialOverridePtr = nullptr;
        if (object.material) {
            materialOverride = toMaterial(*object.material);
            materialOverridePtr = &materialOverride;
        }
        SceneImporter::ImportObjScene(scene, object.path, materialOverridePtr);
        if (SceneObject* root = scene.getActiveObject())
            root->setLocalTransform(transform);
    } else {
        const MaterialX::DocumentPtr materialDocument = object.material
            ? nr::materialx::documentFromAuthoring(toMaterial(*object.material))
            : nr::materialx::defaultMaterial();

        // Mark the material as needing MaterialX compilation when a path is
        // specified. The actual compilation happens later (in runCli()), once
        // the Raytracer exists and can register OptiX modules.
        MeshAssetRef meshAsset;
        if (object.type == "cube")
            meshAsset = scene.add(MeshAsset::CreateCube(scene, object.name, materialDocument));
        else if (object.type == "plane")
            meshAsset = scene.add(MeshAsset::CreatePlane(scene, object.name, materialDocument));
        else if (object.type == "disk")
            meshAsset = scene.add(MeshAsset::CreateDisk(scene, object.name, materialDocument));
        else if (object.type == "sphere")
            meshAsset = scene.add(MeshAsset::CreateSphere(scene, object.name, materialDocument));
        else
            throw std::runtime_error("Unknown scene object type: " + object.type);

        if (object.material && !object.material->materialx_path.empty())
        {
            const uint32_t matIndex = meshAsset.get()->getMaterialIds()[0];
            // Ensure the vector is large enough and store the path. The path
            // is the source of truth for disk-backed materials: the XML is
            // only read from it at compile time (import), never retained in
            // memory. The in-memory document slot is cleared so the path
            // takes precedence.
            auto& paths = scene.getMaterialXSourcePaths();
            if (paths.size() <= matIndex)
                paths.resize(matIndex + 1);
            paths[matIndex] = object.material->materialx_path;
            auto& documents = scene.getMaterialXDocuments();
            if (documents.size() <= matIndex)
                documents.resize(matIndex + 1);
            documents[matIndex] = nullptr;
        }

        scene.add(std::make_unique<MeshInstance>(scene, object.name, meshAsset, transform));
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
    constexpr glz::opts readOptions{.error_on_unknown_keys = true};
    if (const auto error = glz::read<readOptions>(file, json))
        throw std::runtime_error("Failed to parse scene JSON: " + glz::format_error(error, json));

    scene.clear();

    const size_t proceduralMeshCount = static_cast<size_t>(
        std::ranges::count_if(file.objects,
            [](const nr::sceneio::ObjectFile& object) {
                return object.type == "cube" || object.type == "plane"
                    || object.type == "disk" || object.type == "sphere";
            }));
    scene.reserveForImport(
        proceduralMeshCount, proceduralMeshCount, file.objects.size());

    if (file.environment) {
        Environment& environment = scene.getEnvironment();
        environment.color = toVec3(file.environment->color);
        environment.lightingExposure = file.environment->lighting_exposure;
        environment.visibleExposure = file.environment->visible_exposure;
        environment.updateDerivedSettings();
    }

    if (file.render_settings) {
        scene.getRenderSettings().maxSamples = file.render_settings->max_samples;
        scene.getRenderSettings().noiseLimitEnabled = file.render_settings->noise_limit_enabled;
        scene.getRenderSettings().noiseLevel = std::max(file.render_settings->noise_level, 0.0f);
        scene.getRenderSettings().aovEnabled = file.render_settings->aov_enabled;
        scene.getRenderSettings().optixDenoiserEnabled =
            file.render_settings->optix_denoiser_enabled;
        scene.getRenderSettings().optixDenoiserMinSamples =
            std::max(file.render_settings->optix_denoiser_min_samples, 1);
        scene.getRenderSettings().gaussianShadingMode =
            file.render_settings->gaussian_shading_mode == static_cast<int>(GaussianShadingMode::DirectColor)
                ? GaussianShadingMode::DirectColor
                : GaussianShadingMode::GlobalIllumination;
        scene.getRenderSettings().gaussianRenderSphericalHarmonics =
            clampSphericalHarmonicsOrder(file.render_settings->gaussian_render_sh_degree);
    }

    const size_t activeCameraCount = std::ranges::count_if(file.objects,
        [](const nr::sceneio::ObjectFile& object) {
            return object.type == "camera" && object.camera && object.camera->active;
        });
    if (activeCameraCount > 1)
        throw std::runtime_error("Scene contains more than one active camera");

    for (const nr::sceneio::ObjectFile& object : file.objects)
        addObject(scene, object);

    if (activeCameraCount == 0)
        scene.setActiveCamera(nullptr);
    scene.reclaimUnusedResources();
}
