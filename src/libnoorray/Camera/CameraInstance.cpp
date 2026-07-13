#include "CameraInstance.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <imgui.h>
#include "Camera/RealisticCamera.h"
#include "Camera/RossPsfCamera.h"
#include "CUDA/ManagedMemory.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/quaternion.hpp"
#include "Scene/Scene.h"
#include "UI/ImGuiManager.h"

// ── unified-memory allocation ─────────────────────────────────────────────────

void CameraInstance::allocateCamera(CameraProjectionType type)
{
    Camera state = camera->cloneBaseState();
    std::unique_ptr<Sensor> transferredSensor = camera->releaseSensor();
    std::string sharedLensPath;
    std::string sharedGlassCatalogPaths;
    if (const auto* realistic = camera->CastOrNullptr<RealisticCamera>()) {
        sharedLensPath = realistic->getLensPath();
        sharedGlassCatalogPaths = realistic->getGlassCatalogPaths();
    } else if (const auto* hybrid = camera->CastOrNullptr<RossPsfCamera>()) {
        sharedLensPath = hybrid->getLensPath();
        sharedGlassCatalogPaths = hybrid->getGlassCatalogPaths();
    }
    state.sensor = Sensor(nullptr);

    camera.reset(Camera::create(type, std::move(transferredSensor)).release());
    static_cast<Camera&>(*camera) = state;
    tagCamera();

    if (auto* realistic = camera->CastOrNullptr<RealisticCamera>())
        realistic->setOpticsPaths(sharedLensPath, sharedGlassCatalogPaths);
    else if (auto* hybrid = camera->CastOrNullptr<RossPsfCamera>())
        hybrid->setOpticsPaths(sharedLensPath, sharedGlassCatalogPaths);

}

// ── constructor / destructor ──────────────────────────────────────────────────

void CameraInstance::tagCamera()
{
    if (auto* concrete = dynamic_cast<PerspectiveCamera*>(camera.get()))
        static_cast<TaggedCamera&>(*camera) = TaggedCamera(concrete);
    else if (auto* concrete = dynamic_cast<ThinLensCamera*>(camera.get()))
        static_cast<TaggedCamera&>(*camera) = TaggedCamera(concrete);
    else if (auto* concrete = dynamic_cast<OrthographicCamera*>(camera.get()))
        static_cast<TaggedCamera&>(*camera) = TaggedCamera(concrete);
    else if (auto* concrete = dynamic_cast<FisheyeCamera*>(camera.get()))
        static_cast<TaggedCamera&>(*camera) = TaggedCamera(concrete);
    else if (auto* concrete = dynamic_cast<RealisticCamera*>(camera.get()))
        static_cast<TaggedCamera&>(*camera) = TaggedCamera(concrete);
    else if (auto* concrete = dynamic_cast<RossPsfCamera*>(camera.get()))
        static_cast<TaggedCamera&>(*camera) = TaggedCamera(concrete);
    else
        throw std::invalid_argument("CameraInstance requires a concrete Camera type");
}

CameraInstance::CameraInstance(
    std::unique_ptr<Camera> ownedCamera, const std::string& name, Transform transform)
    : SceneObject(name, transform), camera(ownedCamera.release())
{
    if (!camera)
        throw std::invalid_argument("CameraInstance requires a Camera");
    tagCamera();
    rebuildCamera();
}

CameraInstance::CameraInstance(const CameraInstance& other)
    : SceneObject(other),
      arcballPivot(other.arcballPivot),
      arcballMode(other.arcballMode)
{
    nr::synchronizeBeforeManagedMutation("Camera clone");
    other.camera->DispatchCPU([&](const auto* source) {
        using CameraType = std::remove_cvref_t<decltype(*source)>;
        camera.reset(new CameraType(*source));
    });
    tagCamera();
    rebuildCamera();
}

CameraInstance::~CameraInstance() = default;

// ── core ──────────────────────────────────────────────────────────────────────

void CameraInstance::markDirty()
{
    if (!scene) return;
    scene->setDirtyFlag(CameraState);
    scene->setDirtyFlag(Accumulation);
}

void CameraInstance::rebuildCamera()
{
    const quat  rot    = getRotation();
    const vec3  dir    = normalize(rot * LocalForward);
    const vec3  up     = normalize(rot * LocalUp);
    const vec3  right  = normalize(cross(dir, up));

    const mat4 cameraToWorld = mat4(
        vec4(right,         0.f),
        vec4(up,            0.f),
        vec4(-dir,          0.f),
        vec4(getPosition(), 1.f));

    camera->setCameraToWorld(cameraToWorld);
}

void CameraInstance::onTransformUpdated()
{
    SceneObject::onTransformUpdated();
    rebuildCamera();
    if (scene) scene->setDirtyFlag(CameraState);
}

void CameraInstance::switchTo(CameraProjectionType type)
{
    if (getProjectionType() == type)
        return;
    allocateCamera(type);
    rebuildCamera();
}

std::unique_ptr<SceneObject> CameraInstance::clone() const
{
    return std::make_unique<CameraInstance>(*this);
}

// ── queries ──────────────────────────────────────────────────────────────────

CameraProjectionType CameraInstance::getProjectionType() const
{
    if (camera->Is<ThinLensCamera>())    return CameraProjectionType::ThinLens;
    if (camera->Is<OrthographicCamera>()) return CameraProjectionType::Orthographic;
    if (camera->Is<FisheyeCamera>())     return CameraProjectionType::Fisheye;
    if (camera->Is<RealisticCamera>())   return CameraProjectionType::Realistic;
    if (camera->Is<RossPsfCamera>())     return CameraProjectionType::HybridPsf;
    return CameraProjectionType::Perspective;
}

const char* CameraInstance::getProjectionName() const
{
    if (camera->Is<ThinLensCamera>())    return "Thin Lens";
    if (camera->Is<OrthographicCamera>()) return "Orthographic";
    if (camera->Is<FisheyeCamera>())     return "Fisheye";
    if (camera->Is<RealisticCamera>())   return "Realistic";
    if (camera->Is<RossPsfCamera>())     return "Hybrid PSF";
    return "Perspective";
}

mat4 CameraInstance::getViewMatrix() const
{
    const vec3 dir = normalize(getRotation() * LocalForward);
    const vec3 up  = normalize(getRotation() * LocalUp);
    return lookAt(getPosition(), getPosition() + dir, up);
}

mat4 CameraInstance::getProjectionMatrix() const
{
    const Sensor& sensor = camera->getSensor();
    const float aspect = sensor.aspectRatio();
    const float focalLength = camera->getFocalLength();
    const float fovY = 2.f * std::atan(sensor.height() / (2.f * focalLength));
    return perspective(fovY, aspect, 0.01f, 10000.f);
}

// ── arcball ──────────────────────────────────────────────────────────────────

void CameraInstance::setArcballPivot(const vec3& pivot)
{
    arcballPivot = pivot;
    const float distance = std::max(0.001f, glm::distance(getPosition(), pivot));
    camera->setFocusDistance(distance);
}

// ── update (input) ───────────────────────────────────────────────────────────

void CameraInstance::update(const float dx, const float dy)
{
    const vec3 oldPosition = getPosition();
    const quat oldRotation = getRotation();

    ImGuiIO& io = ImGui::GetIO();

    if (arcballMode) {
        vec3 position    = getPosition();
        quat orientation = getRotation();
        float moveSpeed  = io.DeltaTime * 5.f;
        if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) moveSpeed *= 10.f;

        const vec3 dirToCamera = normalize(position - arcballPivot);
        if (ImGui::IsKeyDown(ImGuiKey_W)) position -= dirToCamera * moveSpeed;
        if (ImGui::IsKeyDown(ImGuiKey_S)) position += dirToCamera * moveSpeed;

        constexpr float sensitivity = 0.004f;
        const quat yawQuat   = angleAxis(dx * sensitivity, WorldUp);
        const quat pitchQuat = angleAxis(dy * sensitivity, orientation * LocalRight);
        vec3 offset = position - arcballPivot;
        offset = yawQuat * pitchQuat * offset;
        setPosition(arcballPivot + offset);
        setRotation(normalize(yawQuat * pitchQuat * orientation));
    } else {
        constexpr float sensitivity = 0.1f;
        const float yaw   = radians(-dx * sensitivity);
        const float pitch = radians(-dy * sensitivity);
        quat rot = getRotation();
        const quat yawQuat   = angleAxis(yaw, WorldUp);
        const quat pitchQuat = angleAxis(pitch, rot * LocalRight);
        const quat newRot    = normalize(yawQuat * pitchQuat * rot);
        setRotation(newRot);

        float speed = io.DeltaTime * 5.f;
        if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) speed *= 10.f;

        vec3 position   = getPosition();
        const vec3 fwd  = normalize(newRot * LocalForward);
        const vec3 up   = normalize(newRot * LocalUp);
        const vec3 rgt  = normalize(cross(fwd, up));

        if (ImGui::IsKeyDown(ImGuiKey_W)) position += fwd * speed;
        if (ImGui::IsKeyDown(ImGuiKey_S)) position -= fwd * speed;
        if (ImGui::IsKeyDown(ImGuiKey_A)) position -= rgt * speed;
        if (ImGui::IsKeyDown(ImGuiKey_D)) position += rgt * speed;
        if (ImGui::IsKeyDown(ImGuiKey_E)) position += up  * speed;
        if (ImGui::IsKeyDown(ImGuiKey_Q)) position -= up  * speed;
        setPosition(position);
    }

    if (oldPosition != getPosition() || oldRotation != getRotation())
        markDirty();
}

// ── UI ────────────────────────────────────────────────────────────────────────

bool CameraInstance::renderUi()
{
    const bool instanceChanged = SceneObject::renderUi();

    ImGuiManager::tableRowLabel("Camera");
    const std::string cameraLabel = std::string(getProjectionName()) + "###CameraProperties";
    if (!ImGui::TreeNodeEx(cameraLabel.c_str(), ImGuiTreeNodeFlags_Framed))
        return instanceChanged;

    bool changed = false;
    if (ImGui::BeginTable("CameraTable", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGuiManager::tableRowLabel("Type");
        static const CameraProjectionType projectionTypes[] = {
            CameraProjectionType::Perspective, CameraProjectionType::ThinLens,
            CameraProjectionType::Realistic,   CameraProjectionType::HybridPsf,
            CameraProjectionType::Orthographic, CameraProjectionType::Fisheye,
        };
        static const char* projectionNames[] = {
            "Perspective", "Thin Lens", "Realistic", "Hybrid PSF", "Orthographic", "Fisheye"};
        constexpr int projectionCount = 6;
        int projectionIndex = 0;
        for (int i = 0; i < projectionCount; ++i)
            if (projectionTypes[i] == getProjectionType()) { projectionIndex = i; break; }
        if (ImGui::Combo("##CameraProjection", &projectionIndex, projectionNames, projectionCount)) {
            switchTo(projectionTypes[projectionIndex]);
            changed = true;
        }

        changed |= camera->DispatchCPU([](auto* cam) { return cam->renderUi(); });
        SensorType requestedSensorType{};
        if (camera->getSensor().consumeRequestedType(requestedSensorType)) {
            const Sensor& currentSensor = camera->getSensor();
            if (requestedSensorType == SensorType::ScatterPsf)
                camera->setSensor(std::make_unique<ScatterPsfSensor>(currentSensor));
            else if (requestedSensorType == SensorType::GatherPsf)
                camera->setSensor(std::make_unique<GatherPsfSensor>(currentSensor));
            else
                camera->setSensor(std::make_unique<RectangularSensor>(currentSensor));
            changed = true;
        }
        ImGui::EndTable();
    }
    ImGui::TreePop();

    if (changed) {
        if (scene) scene->setDirtyFlag(Accumulation);
    }
    return instanceChanged || changed;
}
