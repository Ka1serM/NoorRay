#include "PerspectiveCamera.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <imgui.h>
#include <cmath>
#include <iostream>

#include "glm/gtx/rotate_vector.hpp"
#include "SDL3/SDL_mouse.h"
#include "UI/ImGuiManager.h"
#include "Scene/Scene.h"

static constexpr auto FRONT = vec3(0, 0, 1);
static constexpr auto WORLD_UP = vec3(0.0f, -1.0f, 0.0f);
static constexpr auto VULKAN_Z_PLUS = vec3(0.0f, 0.0f, 1.0f);

PerspectiveCamera::PerspectiveCamera(Scene& scene, const std::string& name, Transform transform, float aspect, float sensorWidth, float sensorHeight, float focalLength, float aperture, float focusDistance, float bokehBias) 
    : SceneObject(scene, name, transform), scene(scene), aspectRatio(aspect), sensorWidth(sensorWidth), sensorHeight(sensorHeight)
{
    cameraData.focalLength = focalLength;
    cameraData.aperture = aperture;
    cameraData.focusDistance = focusDistance;
    cameraData.bokehBias = bokehBias;

    updateCameraData();
    updateHorizontalVertical();
}

void PerspectiveCamera::updateCameraData() {
    cameraData.position = getPosition();
    cameraData.direction = normalize(getRotation() * VULKAN_Z_PLUS); // Vulkan +Z;
}

void PerspectiveCamera::updateHorizontalVertical() {
    const vec3 direction = cameraData.direction;
    const vec3 right = normalize(cross(direction, WORLD_UP));
    const vec3 up = normalize(cross(right, direction));

    cameraData.horizontal = right * sensorWidth * 0.001f; // Convert mm to meters
    cameraData.vertical = up * sensorWidth / aspectRatio * 0.001f; // Convert mm to meters
}

mat4 PerspectiveCamera::getViewMatrix() const
{
    const vec3 target = getPosition() + cameraData.direction;
    return lookAt(getPosition(), target, WORLD_UP);
}

mat4 PerspectiveCamera::getProjectionMatrix() const {
    float fovX = 2.0f * atan(sensorWidth * 0.5f / cameraData.focalLength);
    float fovY = 2.0f * atan(tan(fovX * 0.5f) / aspectRatio);
    return perspective(fovY, aspectRatio, 0.1f, 1000.0f);
}

void PerspectiveCamera::setFocalLength(const float val) {
    cameraData.focalLength = val;
    updateHorizontalVertical();
   scene.setDirtyFlag(Accumulation);
}

void PerspectiveCamera::setAperture(const float val) {
    cameraData.aperture = val;
   scene.setDirtyFlag(Accumulation);
}

void PerspectiveCamera::setFocusDistance(const float val) {
    cameraData.focusDistance = val;
   scene.setDirtyFlag(Accumulation);
}

void PerspectiveCamera::setBokehBias(const float val) {
    cameraData.bokehBias = val;
   scene.setDirtyFlag(Accumulation);
}

void PerspectiveCamera::setSensorSize(const float width, const float height) {
    sensorWidth = width;
    sensorHeight = height;
    updateHorizontalVertical();
   scene.setDirtyFlag(Accumulation);
}

void PerspectiveCamera::update() {
    const bool wasDirty = scene.isDirty(Accumulation);
    const vec3 oldPosition = getPosition();
    const vec3 oldDirection = cameraData.direction;

    float dx, dy;
    SDL_GetRelativeMouseState(&dx, &dy);
    ImGuiIO& io = ImGui::GetIO();
        
    if (arcballMode) {
        vec3 position = getPosition();
        quat orientation = getRotation();

        // MOVEMENT LOGIC
        float moveSpeed = io.DeltaTime * 5.0f; // Base speed for movement
        if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
            moveSpeed *= 10.0f; // Speed up with Shift key

        // Zoom (W/S): Move along the vector from the pivot to the camera
        const vec3 directionToCamera = normalize(position - arcballPivot);
        if (ImGui::IsKeyDown(ImGuiKey_W))
            position -= directionToCamera * moveSpeed;
        if (ImGui::IsKeyDown(ImGuiKey_S))
            position += directionToCamera * moveSpeed;

        // 2. Calculate the rotation quaternions from mouse input
        constexpr float sensitivity = 0.004f;
        const float yawAngle = -dx * sensitivity;
        const float pitchAngle = -dy * sensitivity;
        
        const quat yawQuat = angleAxis(yawAngle, WORLD_UP);
        const vec3 localRight = orientation * vec3(1, 0, 0);
        const quat pitchQuat = angleAxis(pitchAngle, localRight);

        // 3. Update the camera's position (based on rotation and new zoom/pan)
        vec3 offset = position - arcballPivot;
        offset = yawQuat * pitchQuat * offset;
        setPosition(arcballPivot + offset);
    
        // 4. Update the camera's orientation
        setRotation(normalize(yawQuat * pitchQuat * orientation));
    }
    else {
        constexpr float sensitivity = 0.1f;
        const float yaw = radians(-dx * sensitivity);
        const float pitch = radians(-dy * sensitivity);

        quat rot = getRotation();
        vec3 forward = rot * FRONT;
        vec3 right = normalize(cross(forward, WORLD_UP));
        quat yawQuat = angleAxis(yaw, WORLD_UP);
        quat pitchQuat = angleAxis(pitch, right);
        quat newRot = normalize(pitchQuat * yawQuat * rot);
        setRotation(newRot);

        // Movement
        float speed = io.DeltaTime * 5.0f; // Base speed
        if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
            speed *= 10.0f;

        vec3 position = getPosition();
        forward = newRot * vec3(0, 0, 1);
        right = normalize(cross(forward, WORLD_UP));
        vec3 upDir = WORLD_UP;

        if (ImGui::IsKeyDown(ImGuiKey_W)) position += forward * speed;
        if (ImGui::IsKeyDown(ImGuiKey_S)) position -= forward * speed;
        if (ImGui::IsKeyDown(ImGuiKey_A)) position -= right * speed;
        if (ImGui::IsKeyDown(ImGuiKey_D)) position += right * speed;
        if (ImGui::IsKeyDown(ImGuiKey_E)) position += upDir * speed;
        if (ImGui::IsKeyDown(ImGuiKey_Q)) position -= upDir * speed;

        setPosition(position);
    }

    updateCameraData();

    if (wasDirty || !all(epsilonEqual(oldDirection, cameraData.direction, 0.001f)) || !all(epsilonEqual(oldPosition, getPosition(), 0.001f))) {
        updateHorizontalVertical();
        scene.setDirtyFlag(Accumulation);
    }
}

void PerspectiveCamera::renderUi() {
    SceneObject::renderUi();

    ImGui::SeparatorText("Camera Lens");

    bool anyChanged = false;
    ImGuiManager::dragFloatRow("Focal Length", getFocalLength(), 0.1f, 10.0f, 500.0f, [&](const float v) {
        setFocalLength(v);
        anyChanged = true;
    });

    ImGuiManager::dragFloatRow("Aperture", getAperture(), 0.01f, 0.0f, 16.0f, [&](const float v) {
        setAperture(v);
        anyChanged = true;
    });

    ImGuiManager::dragFloatRow("Focus Distance", getFocusDistance(), 0.01f, 0.01f, 1000.0f, [&](const float v) {
        setFocusDistance(v);
        anyChanged = true;
    });

    ImGuiManager::dragFloatRow("Bokeh Bias", getBokehBias(), 0.01f, 0.0f, 10.0f, [&](const float v) {
        setBokehBias(v);
        anyChanged = true;
    });

    ImGuiManager::dragFloatRow("Sensor Width", getSensorWidth(), 0.1f, 1.0f, 100.0f, [&](const float v) {
        setSensorSize(v, sensorHeight);
        anyChanged = true;
    });

    ImGuiManager::dragFloatRow("Sensor Height", getSensorHeight(), 0.1f, 1.0f, 100.0f, [&](const float v) {
        setSensorSize(sensorWidth, v);
        anyChanged = true;
    });
    
    if (anyChanged)
       scene.setDirtyFlag(Accumulation);
}

void PerspectiveCamera::onTransformUpdated() {
    SceneObject::onTransformUpdated();
    updateCameraData();
    updateHorizontalVertical();
}