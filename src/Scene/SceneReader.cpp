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
#include "CUDA/rstd/Allocator.h"
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

Camera makeCamera(const nr::sceneio::CameraFile& file)
{
    Camera camera;
    if (file.type == "realistic") {
        nr::rstd::allocator<RealisticCamera> allocator;
        RealisticCamera* realistic = allocator.allocate(1);
        allocator.construct(realistic);
        realistic->apertureDiameterMm = file.aperture_diameter;
        camera = Camera(realistic);
    } else if (file.type == "rosspsf") {
        nr::rstd::allocator<RossPsfCamera> allocator;
        RossPsfCamera* rossPsf = allocator.allocate(1);
        allocator.construct(rossPsf);
        rossPsf->apertureDiameterMm = file.aperture_diameter;
        camera = Camera(rossPsf);
    } else if (file.type == "thinlens") {
        nr::rstd::allocator<ThinLensCamera> allocator;
        ThinLensCamera* thinLens = allocator.allocate(1);
        allocator.construct(thinLens);
        thinLens->fStop = file.aperture_diameter;
        thinLens->bokehBias = std::max(0.001f, file.bokeh_bias);
        camera = Camera(thinLens);
    } else if (file.type == "fisheye") {
        nr::rstd::allocator<FisheyeCamera> allocator;
        FisheyeCamera* fisheye = allocator.allocate(1);
        allocator.construct(fisheye);
        fisheye->fStop = file.aperture_diameter;
        fisheye->bokehBias = std::max(0.001f, file.bokeh_bias);
        camera = Camera(fisheye);
    } else if (file.type == "orthographic") {
        nr::rstd::allocator<OrthographicCamera> allocator;
        OrthographicCamera* orthographic = allocator.allocate(1);
        allocator.construct(orthographic);
        camera = Camera(orthographic);
    } else {
        nr::rstd::allocator<PerspectiveCamera> allocator;
        PerspectiveCamera* perspective = allocator.allocate(1);
        allocator.construct(perspective);
        camera = Camera(perspective);
    }

    camera.setFocalLength(file.focal_length);
    camera.setFocusDistance(file.focus_distance);
    camera.getSensor().setImageSensorPath(file.sensor);
    camera.getSensor().setResolution(
        std::max(file.resolution[0], 1u),
        std::max(file.resolution[1], 1u));
    return camera;
}

void addCamera(Scene& scene, const nr::sceneio::CameraFile& file)
{
    Camera camera = makeCamera(file);
    Transform transform{toVec3(file.position), toVec3(file.rotation_euler), toVec3(file.scale)};
    auto cameraInstance = std::make_unique<CameraInstance>(scene, "Camera", transform, camera);
    if (file.type == "realistic" && !file.lens.empty() && !file.sensor.empty())
        cameraInstance->loadRealisticLens(file.lens, file.glass_catalogs);
    else if (file.type == "rosspsf" && !file.lens.empty() && !file.sensor.empty())
        cameraInstance->loadRossPsfCamera(file.lens, file.glass_catalogs, {});
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

    if (file.render_settings)
        scene.getRenderSettings().maxSamples = file.render_settings->max_samples;

    if (file.camera)
        addCamera(scene, *file.camera);

    for (const nr::sceneio::ObjectFile& object : file.objects)
        addObject(scene, object);
}
