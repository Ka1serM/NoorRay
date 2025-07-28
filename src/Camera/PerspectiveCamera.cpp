#include "PerspectiveCamera.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <imgui.h>
#include <cmath>
#include "GLFW/glfw3.h"
#include "glm/gtx/rotate_vector.hpp"
#include "UI/ImGuiManager.h"

static constexpr glm::vec3 FRONT = glm::vec3(0, 0, 1);
static constexpr glm::vec3 WORLD_UP = glm::vec3(0.0f, -1.0f, 0.0f);
static constexpr glm::vec3 VULKAN_Z_PLUS = glm::vec3(0.0f, 0.0f, 1.0f);

PerspectiveCamera::PerspectiveCamera(Scene& scene, const std::string& name, Transform transform, float aspect, float sensorWidth, float sensorHeight, float focalLength, float aperture, float focusDistance, float bokehBias) : SceneObject(scene, name, transform), scene(scene), aspectRatio(aspect), sensorWidth(sensorWidth), sensorHeight(sensorHeight)
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
    cameraData.direction = glm::normalize(getRotation() * VULKAN_Z_PLUS); // Vulkan +Z;
}

void PerspectiveCamera::updateHorizontalVertical() {
    const glm::vec3 direction = cameraData.direction;
    const glm::vec3 right = glm::normalize(glm::cross(direction, WORLD_UP));
    const glm::vec3 up = glm::normalize(glm::cross(right, direction));

    cameraData.horizontal = right *  sensorWidth * 0.001f; // Convert mm to meters
    cameraData.vertical = up * sensorWidth / aspectRatio * 0.001f; // Convert mm to meters
}

void PerspectiveCamera::setFocalLength(const float val) {
    cameraData.focalLength = val;
    updateHorizontalVertical();
    scene.setAccumulationDirty();
}

void PerspectiveCamera::setAperture(const float val) {
    cameraData.aperture = val;
    scene.setAccumulationDirty();
}

void PerspectiveCamera::setFocusDistance(const float val) {
    cameraData.focusDistance = val;
    scene.setAccumulationDirty();
}

void PerspectiveCamera::setBokehBias(const float val) {
    cameraData.bokehBias = val;
    scene.setAccumulationDirty();
}

void PerspectiveCamera::setSensorSize(const float width, const float height) {
    sensorWidth = width;
    sensorHeight = height;
    updateHorizontalVertical();
    scene.setAccumulationDirty();
}

void PerspectiveCamera::update(InputTracker& inputTracker, float deltaTime) {
    const bool rmbHeld = inputTracker.isMouseButtonHeld(GLFW_MOUSE_BUTTON_RIGHT);
    const bool wasDirty = scene.isAccumulationDirty();

    const glm::vec3 oldPosition = getPosition();
    const glm::vec3 oldDirection = cameraData.direction;

    if (rmbHeld) {
        // --- Rotation ---
        double dx, dy;
        inputTracker.getMouseDelta(dx, dy);

        constexpr float sensitivity = 0.1f;
        const float yaw = glm::radians(-static_cast<float>(dx) * sensitivity);
        const float pitch = glm::radians(-static_cast<float>(dy) * sensitivity); // Invert pitch to match natural movement

        glm::quat rot = getRotation();

        // Camera forward (local +Z)
        glm::vec3 forward = rot * FRONT;

        // Camera right axis (local X)
        glm::vec3 right = glm::normalize(glm::cross(forward, WORLD_UP));

        // Apply yaw around WORLD_UP (global up)
        glm::quat yawQuat = glm::angleAxis(yaw, WORLD_UP);

        // Apply pitch around camera’s right axis (local right)
        glm::quat pitchQuat = glm::angleAxis(pitch, right);

        // Combine rotations: pitch * yaw * current rotation
        glm::quat newRot = glm::normalize(pitchQuat * yawQuat * rot);

        setRotation(newRot);

        // --- Movement ---
        float speed = deltaTime;
        if (inputTracker.isKeyHeld(GLFW_KEY_LEFT_SHIFT))
            speed *= 10.0f;

        glm::vec3 position = getPosition();
        forward = newRot * glm::vec3(0, 0, 1);
        right = glm::normalize(glm::cross(forward, WORLD_UP));
        glm::vec3 upDir = WORLD_UP;

        if (inputTracker.isKeyHeld(GLFW_KEY_W))
            position += forward * speed;
        if (inputTracker.isKeyHeld(GLFW_KEY_S))
            position -= forward * speed;
        if (inputTracker.isKeyHeld(GLFW_KEY_A))
            position -= right * speed;
        if (inputTracker.isKeyHeld(GLFW_KEY_D))
            position += right * speed;
        if (inputTracker.isKeyHeld(GLFW_KEY_E))
            position += upDir * speed;
        if (inputTracker.isKeyHeld(GLFW_KEY_Q))
            position -= upDir * speed;

        setPosition(position);
    }

    updateCameraData();

    const bool changed = wasDirty || !glm::all(glm::epsilonEqual(oldDirection, cameraData.direction, 0.001f)) || !glm::all(glm::epsilonEqual(oldPosition, getPosition(), 0.001f));
    if (changed) {
        updateHorizontalVertical();
        scene.setAccumulationDirty();
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
        scene.setAccumulationDirty();
}

//don't update the mesh instance transform, just the SceneObject transform
void PerspectiveCamera::setPosition(const glm::vec3& pos)
{
    SceneObject::setPosition(pos);
}

void PerspectiveCamera::setRotation(const glm::quat& rot)
{
    SceneObject::setRotation(rot);
}

void PerspectiveCamera::setRotationEuler(const glm::vec3& rot)
{
    SceneObject::setRotationEuler(rot);
}

void PerspectiveCamera::setScale(const glm::vec3& scale)
{
    SceneObject::setScale(scale);
}