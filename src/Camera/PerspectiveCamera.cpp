#include "PerspectiveCamera.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <imgui.h>
#include <cmath>
#include <iostream>

#include "glm/gtx/rotate_vector.hpp"
#include "SDL3/SDL_mouse.h"
#include "Scene/Scene.h"

PerspectiveCamera::PerspectiveCamera(Scene& scene, const std::string& name, Transform transform, float aspect, float sensorWidth, float sensorHeight, float focalLength, float aperture, float focusDistance, float bokehBias) 
    : SceneObject(scene, name, transform), aspectRatio(aspect), sensorWidth(sensorWidth), sensorHeight(sensorHeight)
{
    cameraData.focalLength = focalLength;
    cameraData.aperture = aperture;
    cameraData.focusDistance = focusDistance;
    cameraData.bokehBias = bokehBias;

    updateCameraData();
    updateHorizontalVertical();
}

PerspectiveCamera::PerspectiveCamera(const PerspectiveCamera& other)
    : SceneObject(other),
      aspectRatio(other.aspectRatio),
      sensorWidth(other.sensorWidth),
      sensorHeight(other.sensorHeight),
      cameraData(other.cameraData),
      arcballMode(other.arcballMode),
      arcballPivot(other.arcballPivot)
{
    updateCameraData();
    updateHorizontalVertical();
}

std::unique_ptr<SceneObject> PerspectiveCamera::clone() const {
    return std::make_unique<PerspectiveCamera>(*this);
}

void PerspectiveCamera::updateCameraData() {
    cameraData.position = getPosition();
    // The world-space direction is the local-space forward vector transformed by the camera's rotation
    cameraData.direction = normalize(getRotation() * LOCAL_FORWARD);
}

void PerspectiveCamera::updateHorizontalVertical() {
    // --- MODIFIED ---
    // Calculate right and up vectors directly from the quaternion.
    // This makes them dynamic and avoids dependency on a fixed WORLD_UP.
    const quat orientation = getRotation();
    const vec3 right = normalize(orientation * LOCAL_RIGHT);
    const vec3 up = normalize(orientation * LOCAL_UP);

    cameraData.horizontal = right * sensorWidth * 0.001f; // Convert mm to meters
    cameraData.vertical = up * sensorWidth / aspectRatio * 0.001f; // Convert mm to meters
}

mat4 PerspectiveCamera::getViewMatrix() const
{
    // --- MODIFIED ---
    // The 'up' vector for lookAt must now be dynamic, calculated from the camera's rotation.
    // This prevents the view matrix from becoming unstable when looking straight up or down.
    const vec3 dynamicUp = normalize(getRotation() * LOCAL_UP);
    const vec3 target = getPosition() + cameraData.direction;
    return lookAt(getPosition(), target, dynamicUp);
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
        
        float moveSpeed = io.DeltaTime * 5.0f;
        if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
            moveSpeed *= 10.0f;

        const vec3 directionToCamera = normalize(position - arcballPivot);
        if (ImGui::IsKeyDown(ImGuiKey_W))
            position -= directionToCamera * moveSpeed;
        if (ImGui::IsKeyDown(ImGuiKey_S))
            position += directionToCamera * moveSpeed;

        constexpr float sensitivity = 0.004f;
        const float yawAngle = -dx * sensitivity;
        const float pitchAngle = -dy * sensitivity;
        
        const quat yawQuat = angleAxis(yawAngle, WORLD_UP);
        const vec3 localRight = orientation * LOCAL_RIGHT;
        const quat pitchQuat = angleAxis(pitchAngle, localRight);

        vec3 offset = position - arcballPivot;
        offset = yawQuat * pitchQuat * offset;
        setPosition(arcballPivot + offset);
    
        setRotation(normalize(yawQuat * pitchQuat * orientation));
    }
    else {
        // Refactored fly-camera controls to be fully quaternion-based and robust.
        constexpr float sensitivity = 0.1f;
        const float yaw = radians(-dx * sensitivity);
        const float pitch = radians(-dy * sensitivity);

        quat rot = getRotation();
        
        // Yaw around the world's up-axis to keep the horizon stable.
        quat yawQuat = angleAxis(yaw, WORLD_UP);
        
        // Pitch around the camera's local right-axis. This is key to avoiding gimbal lock.
        const vec3 rightDir = rot * LOCAL_RIGHT;
        quat pitchQuat = angleAxis(pitch, rightDir);

        // Combine rotations and apply to the camera. Order matters.
        quat newRot = normalize(yawQuat * pitchQuat * rot);
        setRotation(newRot);

        // Movement
        float speed = io.DeltaTime * 5.0f; // Base speed
        if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
            speed *= 10.0f;

        vec3 position = getPosition();
        
        // Calculate movement vectors directly from the new rotation
        const vec3 forwardDir = newRot * LOCAL_FORWARD;
        const vec3 upDir = newRot * LOCAL_UP;
        const vec3 finalRightDir = normalize(cross(forwardDir, upDir));

        if (ImGui::IsKeyDown(ImGuiKey_W)) position += forwardDir * speed;
        if (ImGui::IsKeyDown(ImGuiKey_S)) position -= forwardDir * speed;
        if (ImGui::IsKeyDown(ImGuiKey_A)) position -= finalRightDir * speed;
        if (ImGui::IsKeyDown(ImGuiKey_D)) position += finalRightDir * speed;
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

void PerspectiveCamera::onTransformUpdated() {
    SceneObject::onTransformUpdated();
    updateCameraData();
    updateHorizontalVertical();
}