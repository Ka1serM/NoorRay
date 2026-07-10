#include "CameraInstance.h"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>
#include <imgui.h>
#include "Camera/RealisticCamera.h"
#include "CUDA/rstd/Allocator.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/quaternion.hpp"
#include "SDL3/SDL_mouse.h"
#include "Scene/Scene.h"
#include "UI/ImGuiManager.h"

// ── unified-memory allocation ─────────────────────────────────────────────────

void CameraInstance::allocateCamera(CameraProjectionType type)
{
    Camera state{};
    if (gpuCamera != nullptr && *gpuCamera)
        state = gpuCamera->DispatchCPU([](const auto* camera) {
            return static_cast<const Camera&>(*camera);
        });
    freeCamera();
    auto allocate = [&]<typename T>() {
        nr::rstd::allocator<T> allocator;
        T* camera = allocator.allocate(1);
        allocator.construct(camera);
        static_cast<Camera&>(*camera) = state;
        *gpuCamera = Camera(camera);
    };
    switch (type) {
    case CameraProjectionType::ThinLens: allocate.template operator()<ThinLensCamera>(); break;
    case CameraProjectionType::Orthographic: allocate.template operator()<OrthographicCamera>(); break;
    case CameraProjectionType::Fisheye: allocate.template operator()<FisheyeCamera>(); break;
    case CameraProjectionType::Realistic: allocate.template operator()<RealisticCamera>(); break;
    case CameraProjectionType::Perspective:
    default: allocate.template operator()<PerspectiveCamera>(); break;
    }
}

void CameraInstance::freeCamera()
{
    if (gpuCamera == nullptr || !*gpuCamera)
        return;
    gpuCamera->DispatchCPU([](auto* cam) {
        using CameraType = std::remove_reference_t<decltype(*cam)>;
        nr::rstd::allocator<CameraType> allocator;
        allocator.destroy(cam);
        allocator.deallocate(cam, 1);
    });
    *gpuCamera = Camera(nullptr);
}

// ── constructor / destructor ──────────────────────────────────────────────────

CameraInstance::CameraInstance(Scene& scene, const std::string& name, Transform transform,
                               Camera camera)
    : SceneObject(scene, name, transform)
{
    nr::rstd::allocator<Camera> allocator;
    gpuCamera = allocator.allocate(1);
    allocator.construct(gpuCamera);
    *gpuCamera = camera;
    rebuildCamera();
}

CameraInstance::CameraInstance(const CameraInstance& other)
    : SceneObject(other),
      arcballPivot(other.arcballPivot),
      arcballMode(other.arcballMode)
{
    nr::rstd::allocator<Camera> allocator;
    gpuCamera = allocator.allocate(1);
    allocator.construct(gpuCamera);
    allocateCamera(other.getProjectionType());
    const Camera state = other.gpuCamera->DispatchCPU([](const auto* camera) {
        return static_cast<const Camera&>(*camera);
    });
    gpuCamera->DispatchCPU([&](auto* camera) { static_cast<Camera&>(*camera) = state; });
    rebuildCamera();
}

CameraInstance::~CameraInstance()
{
    freeCamera();
    nr::rstd::allocator<Camera> allocator;
    allocator.destroy(gpuCamera);
    allocator.deallocate(gpuCamera, 1);
}

// ── core ──────────────────────────────────────────────────────────────────────

void CameraInstance::markDirty()
{
    rebuildCamera();
    scene.setDirtyFlag(Accumulation);
}

void CameraInstance::loadRealisticLens(const std::string& lensPath, const std::string& sensorPath,
                                        const std::string& glassCatalogPaths)
{
    if (auto* rc = gpuCamera->CastOrNullptr<RealisticCamera>())
        rc->load(lensPath, sensorPath, glassCatalogPaths);
    scene.setDirtyFlag(Accumulation);
}

void CameraInstance::setApertureDiameter(float mm)
{
    if (auto* rc = gpuCamera->CastOrNullptr<RealisticCamera>()) {
        rc->apertureDiameterMm = std::max(0.0f, mm);
        rc->loadLensAndSensor();
    }
    scene.setDirtyFlag(Accumulation);
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

    gpuCamera->DispatchCPU([&](auto* cam) {
        cam->cameraToWorld = cameraToWorld;
    });
}

void CameraInstance::onTransformUpdated()
{
    SceneObject::onTransformUpdated();
    rebuildCamera();
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
    if (gpuCamera->Is<ThinLensCamera>())    return CameraProjectionType::ThinLens;
    if (gpuCamera->Is<OrthographicCamera>()) return CameraProjectionType::Orthographic;
    if (gpuCamera->Is<FisheyeCamera>())     return CameraProjectionType::Fisheye;
    if (gpuCamera->Is<RealisticCamera>())   return CameraProjectionType::Realistic;
    return CameraProjectionType::Perspective;
}

const char* CameraInstance::getProjectionName() const
{
    if (gpuCamera->Is<ThinLensCamera>())    return "Thin Lens";
    if (gpuCamera->Is<OrthographicCamera>()) return "Orthographic";
    if (gpuCamera->Is<FisheyeCamera>())     return "Fisheye";
    if (gpuCamera->Is<RealisticCamera>())   return "Realistic";
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
    const Sensor sensor = gpuCamera->DispatchCPU([](const auto* camera) { return camera->sensor; });
    const float aspect = sensor.aspectRatio();
    const float focalLength = gpuCamera->DispatchCPU([](const auto* camera) { return camera->focalLengthMm; });
    const float fovY = 2.f * std::atan(sensor.heightMm / (2.f * focalLength));
    return perspective(fovY, aspect, 0.01f, 10000.f);
}

// ── arcball ──────────────────────────────────────────────────────────────────

void CameraInstance::setArcballPivot(const vec3& pivot)
{
    arcballPivot = pivot;
    const float distance = std::max(0.001f, glm::distance(getPosition(), pivot));
    gpuCamera->DispatchCPU([&](auto* camera) { camera->focusDistance = distance; });
}

// ── update (input) ───────────────────────────────────────────────────────────

void CameraInstance::update()
{
    const vec3 oldPosition = getPosition();
    const quat oldRotation = getRotation();

    float dx = 0.f, dy = 0.f;
    SDL_GetRelativeMouseState(&dx, &dy);
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

    rebuildCamera();
    if (oldPosition != getPosition() || oldRotation != getRotation())
        scene.setDirtyFlag(Accumulation);
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
        // projection type switcher
        ImGuiManager::tableRowLabel("Projection");
        static const CameraProjectionType projectionTypes[] = {
            CameraProjectionType::Perspective, CameraProjectionType::ThinLens,
            CameraProjectionType::Realistic,   CameraProjectionType::Orthographic,
            CameraProjectionType::Fisheye,
        };
        static const char* projectionNames[] = {"Perspective", "Thin Lens", "Realistic", "Orthographic", "Fisheye"};
        constexpr int projectionCount = 5;
        int projectionIndex = 0;
        for (int i = 0; i < projectionCount; ++i)
            if (projectionTypes[i] == getProjectionType()) { projectionIndex = i; break; }
        if (ImGui::Combo("##CameraProjection", &projectionIndex, projectionNames, projectionCount)) {
            switchTo(projectionTypes[projectionIndex]);
            changed = true;
        }

        changed |= gpuCamera->DispatchCPU([](auto* cam) { return cam->renderUi(); });
        ImGui::EndTable();
    }
    ImGui::TreePop();

    if (changed)
        scene.setDirtyFlag(Accumulation);
    return instanceChanged || changed;
}
