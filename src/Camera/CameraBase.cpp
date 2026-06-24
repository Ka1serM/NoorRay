#include "CameraBase.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <imgui.h>
#include "Camera/FisheyeCamera.h"
#include "Camera/OrthographicCamera.h"
#include "Camera/PerspectiveCamera.h"
#include "Camera/RealisticCamera.h"
#include "Camera/ThinLensCamera.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/quaternion.hpp"
#include "SDL3/SDL_mouse.h"
#include "Scene/Scene.h"
#include "UI/ImGuiManager.h"

float CameraSettings::focalLengthForFov(float fovDegrees)
{
    const float halfAngle = glm::radians(std::clamp(fovDegrees, 1.0f, 179.0f)) * 0.5f;
    return SensorWidthMm / (2.0f * std::tan(halfAngle));
}

float CameraSettings::fovForFocalLength(float focalLength)
{
    const float fov = 2.0f * std::atan(SensorWidthMm / (2.0f * std::max(0.001f, focalLength))) * (180.0f / glm::pi<float>());
    return std::clamp(fov, 1.0f, 179.0f);
}

void CameraSettings::setFieldOfView(float value)
{
    fieldOfView = std::clamp(value, 1.0f, 179.0f);
    focalLengthMm = focalLengthForFov(fieldOfView);
}

void CameraSettings::setFocalLength(float value)
{
    focalLengthMm = std::max(0.001f, value);
    fieldOfView = fovForFocalLength(focalLengthMm);
}

CameraBase::CameraBase(Scene& scene, const std::string& name, Transform transform, CameraSettings settings)
    : SceneObject(scene, name, transform), settings(settings)
{
    if (this->settings.focalLengthMm <= 0.001f)
        this->settings.setFieldOfView(this->settings.fieldOfView);
}

CameraBase::CameraBase(const CameraBase& other)
    : SceneObject(other),
      settings(other.settings),
      cameraData(other.cameraData),
      arcballPivot(other.arcballPivot),
      arcballMode(other.arcballMode),
      renderWidth(other.renderWidth),
      renderHeight(other.renderHeight)
{}

void CameraBase::setRenderSize(uint32_t width, uint32_t height)
{
    renderWidth = std::max(1u, width);
    renderHeight = std::max(1u, height);
    rebuildCameraData();
}

void CameraBase::setArcballPivot(const vec3& pivot)
{
    arcballPivot = pivot;
    settings.focusDistance = std::max(0.001f, glm::distance(getPosition(), pivot));
    rebuildCameraData();
}

void CameraBase::rebuildCameraData()
{
    const float aspectRatio = static_cast<float>(std::max(1u, renderWidth)) / static_cast<float>(std::max(1u, renderHeight));
    const quat rotation = getRotation();
    const vec3 direction = normalize(rotation * LocalForward);
    const vec3 up = normalize(rotation * LocalUp);
    const vec3 right = normalize(cross(direction, up));

    cameraData.position = getPosition();
    cameraData.direction = direction;
    cameraData.cameraType = static_cast<int>(getProjectionType());
    cameraData.nearPlane = 0.01f;
    cameraData.farPlane = 10000.0f;

    computeProjectionData(direction, up, right, aspectRatio);
}

void CameraBase::applyDepthOfField(float focalLengthMm)
{
    cameraData.focusDistance = settings.focusDistance;
    cameraData.bokehBias = settings.bokehBias;
    cameraData.aperture = settings.fStop > 0.0f
        ? (focalLengthMm / settings.fStop) * 0.5f * 0.001f
        : 0.0f;
}

void CameraBase::clearDepthOfField()
{
    cameraData.focusDistance = 0.0f;
    cameraData.bokehBias = 0.0f;
    cameraData.aperture = 0.0f;
}

bool CameraBase::isDataEquivalent(const CameraData& other) const
{
    return std::memcmp(&cameraData, &other, sizeof(CameraData)) == 0;
}

void CameraBase::populateRealisticCameraSettings(RealisticCameraSettings& realisticSettings) const
{
    realisticSettings = {};
}

mat4 CameraBase::getViewMatrix() const
{
    const vec3 dynamicUp = normalize(getRotation() * LocalUp);
    return lookAt(getPosition(), getPosition() + cameraData.direction, dynamicUp);
}

mat4 CameraBase::getProjectionMatrix() const
{
    const float aspectRatio = static_cast<float>(std::max(1u, renderWidth)) / static_cast<float>(std::max(1u, renderHeight));
    if (getProjectionType() == CameraProjectionType::Orthographic) {
        const float halfHeight = std::max(cameraData.orthoHeight, 0.001f) * 0.5f;
        return ortho(-halfHeight * aspectRatio, halfHeight * aspectRatio, -halfHeight, halfHeight, cameraData.nearPlane, cameraData.farPlane);
    }
    // Perspective, ThinLens, and Fisheye all use a perspective projection matrix for rasterisation.
    const float fovY = 2.0f * std::atan(std::tan(glm::radians(settings.fieldOfView) * 0.5f) / aspectRatio);
    return perspective(fovY, aspectRatio, cameraData.nearPlane, cameraData.farPlane);
}

void CameraBase::update()
{
    const vec3 oldPosition = getPosition();
    const quat oldRotation = getRotation();

    float dx = 0.0f;
    float dy = 0.0f;
    SDL_GetRelativeMouseState(&dx, &dy);
    ImGuiIO& io = ImGui::GetIO();

    if (arcballMode) {
        vec3 position = getPosition();
        quat orientation = getRotation();
        float moveSpeed = io.DeltaTime * 5.0f;
        if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
            moveSpeed *= 10.0f;

        const vec3 directionToCamera = normalize(position - arcballPivot);
        if (ImGui::IsKeyDown(ImGuiKey_W))
            position -= directionToCamera * moveSpeed;
        if (ImGui::IsKeyDown(ImGuiKey_S))
            position += directionToCamera * moveSpeed;

        constexpr float sensitivity = 0.004f;
        const quat yawQuat = angleAxis(-dx * sensitivity, WorldUp);
        const quat pitchQuat = angleAxis(-dy * sensitivity, orientation * LocalRight);
        vec3 offset = position - arcballPivot;
        offset = yawQuat * pitchQuat * offset;
        setPosition(arcballPivot + offset);
        setRotation(normalize(yawQuat * pitchQuat * orientation));
    } else {
        constexpr float sensitivity = 0.1f;
        const float yaw = radians(-dx * sensitivity);
        const float pitch = radians(-dy * sensitivity);
        quat rot = getRotation();
        const quat yawQuat = angleAxis(yaw, WorldUp);
        const quat pitchQuat = angleAxis(pitch, rot * LocalRight);
        const quat newRot = normalize(yawQuat * pitchQuat * rot);
        setRotation(newRot);

        float speed = io.DeltaTime * 5.0f;
        if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
            speed *= 10.0f;

        vec3 position = getPosition();
        const vec3 forward = normalize(newRot * LocalForward);
        const vec3 up = normalize(newRot * LocalUp);
        const vec3 right = normalize(cross(forward, up));

        if (ImGui::IsKeyDown(ImGuiKey_W)) position += forward * speed;
        if (ImGui::IsKeyDown(ImGuiKey_S)) position -= forward * speed;
        if (ImGui::IsKeyDown(ImGuiKey_A)) position -= right * speed;
        if (ImGui::IsKeyDown(ImGuiKey_D)) position += right * speed;
        if (ImGui::IsKeyDown(ImGuiKey_E)) position += up * speed;
        if (ImGui::IsKeyDown(ImGuiKey_Q)) position -= up * speed;
        setPosition(position);
    }

    rebuildCameraData();
    if (oldPosition != getPosition() || oldRotation != getRotation())
        scene.setDirtyFlag(Accumulation);
}

void CameraBase::renderUi()
{
    SceneObject::renderUi();

    bool anyChanged = false;
    ImGuiManager::tableRowLabel("Projection");
    static const CameraProjectionType projectionTypes[] = {
        CameraProjectionType::Perspective,
        CameraProjectionType::ThinLens,
        CameraProjectionType::Realistic,
        CameraProjectionType::Orthographic,
        CameraProjectionType::Fisheye,
    };
    static const char* projectionNames[] = {"Perspective", "Thin Lens", "Realistic", "Orthographic", "Fisheye"};
    constexpr int projectionCount = 5;
    int projectionIndex = 0;
    for (int i = 0; i < projectionCount; ++i)
        if (projectionTypes[i] == getProjectionType()) { projectionIndex = i; break; }
    if (ImGui::Combo("##CameraProjection", &projectionIndex, projectionNames, projectionCount)) {
        if (auto replacement = recreateAs(projectionTypes[projectionIndex]))
            scene.replaceObject(this, std::move(replacement));
        return;
    }

    ImGuiManager::dragFloatRow("Field of View", settings.fieldOfView, 0.1f, 1.0f, 179.0f, [&](float v) {
        settings.setFieldOfView(v);
        anyChanged = true;
    });
    ImGuiManager::dragFloatRow("Focal Length", settings.focalLengthMm, 0.1f, 0.001f, 500.0f, [&](float v) {
        settings.setFocalLength(v);
        anyChanged = true;
    });

    if (getProjectionType() != CameraProjectionType::Orthographic) {
        ImGuiManager::dragFloatRow("F-Stop", settings.fStop, 0.1f, 0.0f, 32.0f, [&](float v) {
            settings.fStop = std::max(0.0f, v);
            anyChanged = true;
        });
        ImGuiManager::dragFloatRow("Focus Distance", settings.focusDistance, 0.1f, 0.001f, 1000.0f, [&](float v) {
            settings.focusDistance = std::max(0.001f, v);
            anyChanged = true;
        });
        ImGuiManager::dragFloatRow("Bokeh Bias", settings.bokehBias, 0.01f, 0.001f, 10.0f, [&](float v) {
            settings.bokehBias = std::max(0.001f, v);
            anyChanged = true;
        });
    }

    if (anyChanged) {
        rebuildCameraData();
        scene.setDirtyFlag(Accumulation);
    }
}

void CameraBase::onTransformUpdated()
{
    SceneObject::onTransformUpdated();
    rebuildCameraData();
}

std::unique_ptr<CameraBase> CameraBase::recreateAs(CameraProjectionType type) const
{
    switch (type) {
    case CameraProjectionType::ThinLens:
        return std::make_unique<ThinLensCamera>(scene, name, getTransform(), settings);
    case CameraProjectionType::Realistic:
        return std::make_unique<RealisticCamera>(scene, name, getTransform(), settings);
    case CameraProjectionType::Orthographic:
        return std::make_unique<OrthographicCamera>(scene, name, getTransform(), settings);
    case CameraProjectionType::Fisheye:
        return std::make_unique<FisheyeCamera>(scene, name, getTransform(), settings);
    case CameraProjectionType::Perspective:
    default:
        return std::make_unique<PerspectiveCamera>(scene, name, getTransform(), settings);
    }
}
