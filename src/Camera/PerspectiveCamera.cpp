#include "PerspectiveCamera.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <imgui.h>

#include <cmath>
#include "GLFW/glfw3.h"
#include "glm/gtx/rotate_vector.hpp"

#include "Scene/Scene.h"
#include "UI/ImGuiManager.h"

static constexpr glm::vec3 FRONT = glm::vec3(0, 0, 1);
static constexpr glm::vec3 WORLD_UP = glm::vec3(0.0f, -1.0f, 0.0f);

PerspectiveCamera::PerspectiveCamera(
    Renderer& renderer,
    const std::string& name,
    Transform transform,
    float aspect,
    float sensorWidth,
    float sensorHeight,
    float focalLength,
    float aperture,
    float focusDistance
) : SceneObject(renderer, name, transform), aspectRatio(aspect), sensorWidth(sensorWidth), sensorHeight(sensorHeight) {

    cameraData.focalLength = focalLength;
    cameraData.aperture = aperture;
    cameraData.focusDistance = focusDistance;

    updateCameraData();
    updateHorizontalVertical();
}

glm::vec3 PerspectiveCamera::calculateDirection() const {
    return glm::normalize(transform.getRotation() * glm::vec3(0.0f, 0.0f, 1.0f)); // Vulkan +Z
}

void PerspectiveCamera::updateCameraData() {
    cameraData.position = transform.getPosition();
    cameraData.direction = calculateDirection();
}

void PerspectiveCamera::updateHorizontalVertical() {
    const float sensorWidthMeters = sensorWidth * 0.001f;
    const float sensorHeightMeters = sensorHeight * 0.001f;

    const glm::vec3 direction = cameraData.direction;
    const glm::vec3 right = glm::normalize(glm::cross(direction, WORLD_UP));
    const glm::vec3 up = glm::normalize(glm::cross(right, direction));

    cameraData.horizontal = right * sensorWidthMeters;
    cameraData.vertical = up * sensorHeightMeters;
}

void PerspectiveCamera::setFocalLength(const float val) {
    cameraData.focalLength = val;
    updateHorizontalVertical();
    renderer.markDirty();
}

void PerspectiveCamera::setAperture(const float val) {
    cameraData.aperture = val;
    renderer.markDirty();
}

void PerspectiveCamera::setFocusDistance(const float val) {
    cameraData.focusDistance = val;
    renderer.markDirty();
}

void PerspectiveCamera::setSensorSize(const float width, const float height) {
    sensorWidth = width;
    sensorHeight = height;
    updateHorizontalVertical();
    renderer.markDirty();
}

void PerspectiveCamera::update(InputTracker& inputTracker, float deltaTime) {
    const bool rmbHeld = inputTracker.isMouseButtonHeld(GLFW_MOUSE_BUTTON_RIGHT);
    const bool wasDirty = renderer.getDirty();

    const glm::vec3 oldPosition = transform.getPosition();
    const glm::vec3 oldDirection = cameraData.direction;

    if (rmbHeld) {
        // --- Rotation ---
        double dx, dy;
        inputTracker.getMouseDelta(dx, dy);

        constexpr float sensitivity = 0.1f;
        const float yaw = glm::radians(-static_cast<float>(dx) * sensitivity);
        const float pitch = glm::radians(-static_cast<float>(dy) * sensitivity); // Invert pitch to match natural movement

        glm::quat rot = transform.getRotation();

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

        transform.setRotation(newRot);


        // --- Movement ---
        float speed = 20.0f * deltaTime;
        if (inputTracker.isKeyHeld(GLFW_KEY_LEFT_SHIFT))
            speed *= 10.0f;

        glm::vec3 position = transform.getPosition();
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

        transform.setPosition(position);
    }

    updateCameraData();

    const bool changed =
        !glm::all(glm::epsilonEqual(oldDirection, cameraData.direction, 0.001f)) ||
        !glm::all(glm::epsilonEqual(oldPosition, transform.getPosition(), 0.001f)) ||
        wasDirty;

    if (changed) {
        updateHorizontalVertical();
        renderer.markDirty();
    }
}


void PerspectiveCamera::renderUi() {
    SceneObject::renderUi();

    ImGui::SeparatorText("Camera Lens");

    ImGuiManager::dragFloatRow("Focal Length", getFocalLength(), 0.1f, 10.0f, 500.0f, [&](const float v) {
        setFocalLength(v);
    });

    ImGuiManager::dragFloatRow("Aperture", getAperture(), 0.05f, 0.1f, 16.0f, [&](const float v) {
        setAperture(v);
    });

    ImGuiManager::dragFloatRow("Focus Distance", getFocusDistance(), 0.1f, 0.01f, 1000.0f, [&](const float v) {
        setFocusDistance(v);
    });

    ImGuiManager::dragFloatRow("Sensor Width", getSensorWidth(), 0.1f, 1.0f, 100.0f, [&](const float v) {
        setSensorSize(v, sensorHeight);
    });

    ImGuiManager::dragFloatRow("Sensor Height", getSensorHeight(), 0.1f, 1.0f, 100.0f, [&](const float v) {
        setSensorSize(sensorWidth, v);
    });
}
