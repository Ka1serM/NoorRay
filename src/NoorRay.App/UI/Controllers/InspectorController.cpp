#include "NoorRay.App/UI/Controllers/InspectorController.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <string_view>
#include <variant>

#include "NoorRay.App/UI/Controls/FieldCommit.h"
#include "NoorRay.App/UI/Controls/ViewportElement.h"
#include "NoorRay.App/UI/Models/AppModel.h"
#include "RmlKit/UI/Models/Model.h"

namespace NoorRay::App {

namespace {

constexpr float degrees = 180.0f / Core::pi;
constexpr float millimetres = 1000.0f;
constexpr std::array<std::string_view, 6> lightTypeNames{"point", "spot", "rect", "disk", "sphere", "directional"};
constexpr std::array<std::string_view, 4> cameraTypeNames{"perspective", "orthographic", "fisheye", "realistic"};

template <class Enum, std::size_t count>
Rml::String nameOf(const std::array<std::string_view, count>& names, const Enum value) {
    return Rml::String(names[static_cast<std::size_t>(value)]);
}

template <class Enum, std::size_t count>
Enum valueOf(const std::array<std::string_view, count>& names, const Rml::String& name, const Enum current) {
    const auto found = std::ranges::find(names, std::string_view(name));
    return found == names.end() ? current : static_cast<Enum>(found - names.begin());
}

struct Trs {
    Core::float3 translation;
    Core::float3 eulerRadians;
    Core::float3 scale;
};

//! Euler angles in the order X, then Y, then Z: R = Rz * Ry * Rx.
Trs decompose(const Core::Transform& transform) {
    Core::float3 scale{Core::length(transform.getColumn(0)), Core::length(transform.getColumn(1)), Core::length(transform.getColumn(2))};
    if (transform.determinant() < 0.0f)
        scale.x = -scale.x;
    const auto r = [&](const int row, const int column) {
        const float s = column == 0 ? scale.x : column == 1 ? scale.y : scale.z;
        return s != 0.0f ? transform.rows[row][column] / s : 0.0f;
    };
    const Core::float3 euler{std::atan2(r(2, 1), r(2, 2)), std::asin(std::clamp(-r(2, 0), -1.0f, 1.0f)), std::atan2(r(1, 0), r(0, 0))};
    return {transform.getTranslation(), euler, scale};
}

Core::Quaternion fromEuler(const Core::float3 radians) {
    const float cx = std::cos(radians.x * 0.5f), sx = std::sin(radians.x * 0.5f);
    const float cy = std::cos(radians.y * 0.5f), sy = std::sin(radians.y * 0.5f);
    const float cz = std::cos(radians.z * 0.5f), sz = std::sin(radians.z * 0.5f);
    return {sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz, cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz};
}

Rml::String formatCount(const std::uint64_t count) {
    if (count >= 1'000'000)
        return std::format("{:.2f} M", static_cast<double>(count) / 1e6);
    if (count >= 1'000)
        return std::format("{:.1f} k", static_cast<double>(count) / 1e3);
    return std::format("{}", count);
}

}

InspectorController::InspectorController(RmlKit::UI::Model& model, AppModel& app, Session::Session& session, ViewportElementInstancer& viewports)
    : model(model), app(app), session(session), viewports(viewports) {
    model.bindEvent("inspectorApply", &InspectorController::onApply, this);
    model.bindEvent("environmentApply", &InspectorController::onApplyEnvironment, this);
    model.bindEvent("environmentClear", &InspectorController::onClearEnvironment, this);
    model.bindEvent("inspectorMakeActiveCamera", &InspectorController::onMakeActiveCamera, this);
}

void InspectorController::showTransform(const Core::Transform& transform) {
    const Trs trs = decompose(transform);
    InspectorModel& inspector = app.inspector;
    inspector.positionX = trs.translation.x;
    inspector.positionY = trs.translation.y;
    inspector.positionZ = trs.translation.z;
    inspector.rotationX = trs.eulerRadians.x * degrees;
    inspector.rotationY = trs.eulerRadians.y * degrees;
    inspector.rotationZ = trs.eulerRadians.z * degrees;
    inspector.scaleX = trs.scale.x;
    inspector.scaleY = trs.scale.y;
    inspector.scaleZ = trs.scale.z;
}

Core::Transform InspectorController::readTransform() const {
    const InspectorModel& inspector = app.inspector;
    const Core::float3 euler{inspector.rotationX / degrees, inspector.rotationY / degrees, inspector.rotationZ / degrees};
    return Core::Transform::fromTrs({inspector.positionX, inspector.positionY, inspector.positionZ}, fromEuler(euler),
                                    {inspector.scaleX, inspector.scaleY, inspector.scaleZ});
}

void InspectorController::showMaterials(const Scene::Instance& instance, const Scene::Mesh* mesh) {
    const Scene::Scene& scene = session.getScene();
    std::vector<MaterialLink>& links = app.inspector.materials;
    const auto add = [&](const Scene::MaterialHandle handle) {
        const Scene::Material* material = scene.find(handle);
        const int index = static_cast<int>(scene.getMaterials().denseIndex(handle));
        if (material && std::ranges::none_of(links, [&](const MaterialLink& link) { return link.sceneIndex == index; }))
            links.push_back({material->name, index});
    };
    app.inspector.materialOverride = scene.find(instance.material) != nullptr;
    if (app.inspector.materialOverride) {
        add(instance.material);
        return;
    }
    if (mesh)
        for (const Scene::MeshSubset& subset : mesh->subsets)
            add(subset.material);
}

void InspectorController::showSelectedTransform() {
    const Scene::Scene& scene = session.getScene();
    std::visit(
        [&](const auto& handle) {
            if constexpr (!std::is_same_v<std::decay_t<decltype(handle)>, std::monostate>)
                if (const auto* item = scene.find(handle))
                    showTransform(item->transform);
        },
        selection);
    model.dirty("inspector");
}

void InspectorController::show(const Selection& selected) {
    selection = selected;
    InspectorModel& inspector = app.inspector;
    inspector.kind.clear();
    inspector.heading.clear();
    inspector.materials.clear();
    inspector.materialOverride = false;
    const Scene::Scene& scene = session.getScene();
    if (const auto* handle = std::get_if<Scene::InstanceHandle>(&selection)) {
        if (const Scene::Instance* instance = scene.find(*handle)) {
            const Scene::Mesh* mesh = scene.find(instance->mesh);
            inspector.kind = "instance";
            inspector.heading = "Object";
            inspector.name = mesh ? Rml::String(mesh->name) : Rml::String("Instance");
            inspector.meshName = mesh ? Rml::String(mesh->name) : Rml::String();
            inspector.triangles = mesh ? static_cast<int>(mesh->getTriangleCount()) : 0;
            inspector.vertices = mesh ? static_cast<int>(mesh->getVertexCount()) : 0;
            inspector.visible = !Scene::hasFlag(instance->flags, Scene::InstanceFlags::Hidden);
            inspector.doubleSided = Scene::hasFlag(instance->flags, Scene::InstanceFlags::DoubleSided);
            inspector.castsShadows = !Scene::hasFlag(instance->flags, Scene::InstanceFlags::NoShadow);
            inspector.cameraVisible = !Scene::hasFlag(instance->flags, Scene::InstanceFlags::NoCameraRays);
            showTransform(instance->transform);
            showMaterials(*instance, mesh);
        }
    } else if (const auto* handle = std::get_if<Scene::LightHandle>(&selection)) {
        if (const Scene::Light* light = scene.find(*handle)) {
            inspector.kind = "light";
            inspector.heading = "Light";
            inspector.name = light->name;
            inspector.lightType = nameOf(lightTypeNames, light->type);
            inspector.colorR = light->color.x;
            inspector.colorG = light->color.y;
            inspector.colorB = light->color.z;
            inspector.power = light->power;
            inspector.radius = light->radius;
            inspector.width = light->width;
            inspector.height = light->height;
            inspector.spotAngle = light->spotAngle * degrees;
            inspector.spotBlend = light->spotBlend;
            inspector.angle = light->angle * degrees;
            showTransform(light->transform);
        }
    } else if (const auto* handle = std::get_if<Scene::CameraHandle>(&selection)) {
        if (const Scene::Camera* camera = scene.find(*handle)) {
            inspector.kind = "camera";
            inspector.heading = "Camera";
            inspector.name = camera->name;
            inspector.cameraType = nameOf(cameraTypeNames, camera->type);
            inspector.activeCamera = scene.getActiveCamera() == *handle;
            inspector.fieldOfView = camera->verticalFov * degrees;
            inspector.orthographicHeight = camera->orthographicHeight;
            inspector.focusDistance = camera->focusDistance;
            inspector.fStop = camera->fStop;
            inspector.blades = static_cast<int>(camera->blades);
            inspector.sensorWidth = camera->sensor.width * millimetres;
            inspector.sensorHeight = camera->sensor.height * millimetres;
            inspector.lensElements = static_cast<int>(camera->lens.elements.size());
            showTransform(camera->transform);
        }
    } else if (const auto* handle = std::get_if<Scene::GaussianCloudHandle>(&selection)) {
        if (const Scene::GaussianCloud* cloud = scene.find(*handle)) {
            inspector.kind = "gaussians";
            inspector.heading = "Gaussians";
            inspector.name = cloud->name;
            inspector.gaussianCount = formatCount(cloud->getCount());
            inspector.shDegree = static_cast<int>(cloud->shDegree);
            showTransform(cloud->transform);
        }
    } else if (const auto* handle = std::get_if<Scene::VolumeHandle>(&selection)) {
        if (const Scene::Volume* volume = scene.find(*handle)) {
            inspector.kind = "volume";
            inspector.heading = "Volume";
            inspector.name = volume->name;
        }
    }
    model.dirty("inspector");
}

void InspectorController::showEnvironment() {
    const Scene::Environment& environment = session.getScene().getEnvironment();
    const Scene::Texture* texture = session.getScene().find(environment.texture);
    EnvironmentModel& shown = app.environment;
    shown.textureName = texture ? Rml::String(texture->name) : Rml::String();
    shown.colorR = environment.color.x;
    shown.colorG = environment.color.y;
    shown.colorB = environment.color.z;
    shown.strength = environment.strength;
    shown.rotation = environment.rotation * degrees;
    shown.visibleToCamera = environment.visibleToCamera;
    model.dirty("environment");
}

void InspectorController::changed() {
    viewports.invalidate();
    model.dirty("inspector");
}

void InspectorController::onApply(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    if (!FieldCommit::isCommit(event))
        return;
    RmlKit::UI::Model::requireNoArguments(arguments, "inspectorApply");
    Scene::Scene& scene = session.getScene();
    const InspectorModel& inspector = app.inspector;
    if (const auto* handle = std::get_if<Scene::InstanceHandle>(&selection)) {
        const Scene::Instance* instance = scene.find(*handle);
        if (!instance)
            return;
        Scene::InstanceFlags flags = Scene::InstanceFlags::None;
        if (!inspector.visible)
            flags = flags | Scene::InstanceFlags::Hidden;
        if (inspector.doubleSided)
            flags = flags | Scene::InstanceFlags::DoubleSided;
        if (!inspector.castsShadows)
            flags = flags | Scene::InstanceFlags::NoShadow;
        if (!inspector.cameraVisible)
            flags = flags | Scene::InstanceFlags::NoCameraRays;
        if (flags != instance->flags)
            scene.setInstanceFlags(*handle, flags);
        scene.setInstanceTransform(*handle, readTransform());
    } else if (const auto* handle = std::get_if<Scene::LightHandle>(&selection)) {
        const Scene::Light* current = scene.find(*handle);
        if (!current)
            return;
        Scene::Light light = *current;
        light.type = valueOf(lightTypeNames, inspector.lightType, light.type);
        light.color = {std::max<float>(inspector.colorR, 0.0f), std::max<float>(inspector.colorG, 0.0f), std::max<float>(inspector.colorB, 0.0f)};
        light.power = std::max<float>(inspector.power, 0.0f);
        light.radius = std::max<float>(inspector.radius, 0.0f);
        light.width = std::max<float>(inspector.width, 1e-4f);
        light.height = std::max<float>(inspector.height, 1e-4f);
        light.spotAngle = std::clamp<float>(inspector.spotAngle, 0.1f, 179.0f) / degrees;
        light.spotBlend = std::clamp<float>(inspector.spotBlend, 0.0f, 1.0f);
        light.angle = std::clamp<float>(inspector.angle, 0.0f, 90.0f) / degrees;
        light.transform = readTransform();
        scene.setLight(*handle, light);
    } else if (const auto* handle = std::get_if<Scene::CameraHandle>(&selection)) {
        const Scene::Camera* current = scene.find(*handle);
        if (!current)
            return;
        Scene::Camera camera = *current;
        camera.type = valueOf(cameraTypeNames, inspector.cameraType, camera.type);
        camera.verticalFov = std::clamp<float>(inspector.fieldOfView, 1.0f, 179.0f) / degrees;
        camera.orthographicHeight = std::max<float>(inspector.orthographicHeight, 1e-4f);
        camera.focusDistance = std::max<float>(inspector.focusDistance, 1e-4f);
        camera.fStop = std::max<float>(inspector.fStop, 0.0f);
        camera.blades = static_cast<std::uint32_t>(std::max(inspector.blades, 0));
        camera.sensor.width = std::max<float>(inspector.sensorWidth, 0.1f) / millimetres;
        camera.sensor.height = std::max<float>(inspector.sensorHeight, 0.1f) / millimetres;
        camera.previousTransform = camera.transform;
        camera.transform = readTransform();
        scene.setCamera(*handle, camera);
        if (scene.getActiveCamera() == *handle)
            viewports.followActiveCamera();
    } else if (const auto* handle = std::get_if<Scene::GaussianCloudHandle>(&selection)) {
        Scene::GaussianCloud* cloud = scene.findForTraining(*handle);
        if (!cloud)
            return;
        cloud->transform = readTransform();
        scene.markGaussiansChanged();
    }
    changed();
}

void InspectorController::onApplyEnvironment(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    if (!FieldCommit::isCommit(event))
        return;
    RmlKit::UI::Model::requireNoArguments(arguments, "environmentApply");
    Scene::Environment environment = session.getScene().getEnvironment();
    const EnvironmentModel& shown = app.environment;
    environment.color = {std::max<float>(shown.colorR, 0.0f), std::max<float>(shown.colorG, 0.0f), std::max<float>(shown.colorB, 0.0f)};
    environment.strength = std::max<float>(shown.strength, 0.0f);
    environment.rotation = shown.rotation / degrees;
    environment.visibleToCamera = shown.visibleToCamera;
    session.getScene().setEnvironment(environment);
    viewports.invalidate();
    model.dirty("environment");
}

//! Drops the HDRI, so the environment is the uniform color again.
void InspectorController::onClearEnvironment(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "environmentClear");
    Scene::Scene& scene = session.getScene();
    Scene::Environment environment = scene.getEnvironment();
    const Scene::TextureHandle texture = environment.texture;
    environment.texture = {};
    scene.setEnvironment(environment);
    if (scene.find(texture))
        scene.remove(texture);
    viewports.invalidate();
    showEnvironment();
}

void InspectorController::onMakeActiveCamera(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "inspectorMakeActiveCamera");
    const auto* handle = std::get_if<Scene::CameraHandle>(&selection);
    if (!handle)
        return;
    session.getScene().setActiveCamera(*handle);
    viewports.followActiveCamera();
    app.inspector.activeCamera = true;
    changed();
}

}
