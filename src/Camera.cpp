#include "Camera.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <imgui.h>

#include "GLFW/glfw3.h"
#include "glm/gtx/rotate_vector.hpp"

PerspectiveCamera::PerspectiveCamera(
    const glm::vec3& origin,
    const glm::vec3& lookAt,
    float aspect,
    float sensorWidth,   // in mm, e.g. 53.4 for medium format
    float sensorHeight,  // in mm
    float focalLength,   // in mm, e.g. 50
    float fStop,         // e.g. 2.8, 4, 5.6
    float focusDistance  // in meters, e.g. 10.0
) : aspectRatio(aspect), sensorWidth(sensorWidth), sensorHeight(sensorHeight) {

    cameraData.position = origin;
    cameraData.direction = normalize(lookAt - origin);
    cameraData.focalLength = focalLength;
    cameraData.aperture = fStop;  // store f-stop (dimensionless)
    cameraData.focusDistance = focusDistance;
    cameraData.isMoving = false;

    updateHorizontalVertical();
}

void PerspectiveCamera::updateHorizontalVertical() {
    // Convert sensor size and focal length from mm to meters
    float sensorWidthMeters = sensorWidth * 0.001f;
    float sensorHeightMeters = sensorHeight * 0.001f;
    float focalLengthMeters = cameraData.focalLength * 0.001f;

    // Calculate half size of the image plane at focal length distance (no scaling by focusDistance!)
    float halfWidth = sensorWidthMeters * 0.5f;
    float halfHeight = sensorHeightMeters * 0.5f;

    // Camera basis vectors
    glm::vec3 right = glm::normalize(glm::cross(cameraData.direction, UP));
    glm::vec3 up = glm::normalize(glm::cross(right, cameraData.direction));

    // Image plane vectors at focal length distance (focalLengthMeters)
    // scale horizontal/vertical to correct size at focal length distance (pinhole camera model)
    cameraData.horizontal = right * (2.0f * halfWidth);
    cameraData.vertical = up * (2.0f * halfHeight);
}

void PerspectiveCamera::update(InputTracker& inputTracker, float deltaTime) {
    if (!inputTracker.isMouseButtonHeld(GLFW_MOUSE_BUTTON_RIGHT)) {
        cameraData.isMoving = false;
        return;
    }

    glm::vec3 oldDirection = cameraData.direction;
    glm::vec3 oldPosition = cameraData.position;

    double deltaX, deltaY;
    inputTracker.getMouseDelta(deltaX, deltaY);

    float sensitivity = 10.0f * deltaTime;

    glm::vec3 right = glm::normalize(glm::cross(cameraData.direction, UP));

    glm::mat4 yawRot = glm::rotate(glm::mat4(1.0f), glm::radians(-static_cast<float>(deltaX) * sensitivity), UP);
    cameraData.direction = glm::vec3(yawRot * glm::vec4(cameraData.direction, 0.0f));

    right = glm::normalize(glm::cross(cameraData.direction, UP));

    glm::mat4 pitchRot = glm::rotate(glm::mat4(1.0f), glm::radians(-static_cast<float>(deltaY) * sensitivity), right);
    cameraData.direction = glm::normalize(glm::vec3(pitchRot * glm::vec4(cameraData.direction, 0.0f)));

    updateHorizontalVertical();

    float movementSpeed = 20.0f * deltaTime;
    if (inputTracker.isKeyHeld(GLFW_KEY_LEFT_SHIFT))
        movementSpeed *= 10.0f;

    if (inputTracker.isKeyHeld(GLFW_KEY_W))
        cameraData.position += cameraData.direction * movementSpeed;
    if (inputTracker.isKeyHeld(GLFW_KEY_S))
        cameraData.position -= cameraData.direction * movementSpeed;
    if (inputTracker.isKeyHeld(GLFW_KEY_A))
        cameraData.position -= right * movementSpeed;
    if (inputTracker.isKeyHeld(GLFW_KEY_D))
        cameraData.position += right * movementSpeed;
    if (inputTracker.isKeyHeld(GLFW_KEY_E))
        cameraData.position += UP * movementSpeed;
    if (inputTracker.isKeyHeld(GLFW_KEY_Q))
        cameraData.position -= UP * movementSpeed;

    float epsilon = 0.001f;
    cameraData.isMoving =
        !glm::all(glm::epsilonEqual(oldDirection, cameraData.direction, epsilon)) ||
        !glm::all(glm::epsilonEqual(oldPosition, cameraData.position, epsilon));
}

void PerspectiveCamera::setSensorSize(float width, float height) {
    sensorWidth = width;
    sensorHeight = height;
    updateHorizontalVertical();
}

void PerspectiveCamera::setFocalLength(float focalLength) {
    cameraData.focalLength = focalLength;
    updateHorizontalVertical();
}

void PerspectiveCamera::setAperture(float fStop) {
    cameraData.aperture = fStop;
}

void PerspectiveCamera::setFocusDistance(float focusDistance) {
    cameraData.focusDistance = focusDistance;
}

void PerspectiveCamera::renderUI(bool& dirty) {
    ImGui::Begin("Camera");

    ImGui::SeparatorText("Lens");

    if (ImGui::BeginTable("CameraSettings", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableSetupColumn("Label");
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

        auto tableRowDrag = [&](const char* label, const char* id, float& value, float speed, float min, float max, auto setter) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("%s", label);
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (ImGui::DragFloat(id, &value, speed, min, max, "%.3f", ImGuiSliderFlags_AlwaysClamp)) {
                setter(value);
                dirty = true;
            }
        };

        float focal = getFocalLength();
        tableRowDrag("Focal Length", "##FocalLength", focal, 0.1f, 10.0f, 500.0f, [this](float v) {
            setFocalLength(v);
        });

        float aperture = getAperture();
        tableRowDrag("Aperture", "##Aperture", aperture, 0.05f, 0.1f, 16.0f, [this](float v) {
            setAperture(v);
        });

        float focus = getFocusDistance();
        tableRowDrag("Focus Distance", "##FocusDistance", focus, 0.1f, 0.01f, 1000.0f, [this](float v) {
            setFocusDistance(v);
        });

        float width = getSensorWidth();
        tableRowDrag("Sensor Width", "##SensorWidth", width, 0.1f, 1.0f, 100.0f, [this](float v) {
            setSensorSize(v, sensorHeight);
        });

        float height = getSensorHeight();
        tableRowDrag("Sensor Height", "##SensorHeight", height, 0.1f, 1.0f, 100.0f, [this](float v) {
            setSensorSize(sensorWidth, v);
        });

        ImGui::EndTable();
    }

    ImGui::End();
}
