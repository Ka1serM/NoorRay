#include "Camera.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "GLFW/glfw3.h"
#include "glm/gtx/rotate_vector.hpp"

PerspectiveCamera::PerspectiveCamera(const glm::vec3& origin, const glm::vec3& lookAt, const glm::vec3& up, float aspect, float fovDegrees) : UP(up), position(origin) {
    float theta = glm::radians(fovDegrees);
    float halfHeight = tanf(theta * 0.5f);
    float halfWidth = aspect * halfHeight;

    glm::vec3 viewVector = normalize(lookAt - origin); // Forward
    glm::vec3 rightVector = normalize(cross(viewVector, UP));
    glm::vec3 upVector = cross(rightVector, viewVector); // Recompute orthogonal up

    direction = viewVector;
    horizontal = rightVector * halfWidth;
    vertical = upVector * halfHeight;

    //init gpu struct to cam values
    cameraData.position = position;
    cameraData.direction = direction;
    cameraData.horizontal = horizontal;
    cameraData.vertical = vertical;
}

void PerspectiveCamera::update(InputTracker& inputTracker, float deltaTime) {

    if (!inputTracker.isMouseButtonHeld(GLFW_MOUSE_BUTTON_LEFT) && !inputTracker.isMouseButtonHeld(GLFW_MOUSE_BUTTON_RIGHT)) {
        cameraData.isMoving = false;
        return;
    }

    //Detect movement
    glm::vec3 oldDirection = direction;
    glm::vec3 oldPosition = position;

    // Update mouse delta and get it
    double deltaX, deltaY;
    inputTracker.getMouseDelta(deltaX, deltaY);

    float sensitivity = 10.0f * deltaTime;

    //Apply mouse movement (yaw and pitch)
    glm::vec3 right = normalize(cross(direction, UP));         //Right from camera basis
    glm::vec3 up = normalize(cross(right, direction));         //True up vector

    //Apply yaw (around UP)
    glm::mat4 yawRot = glm::rotate(glm::mat4(1.0f), glm::radians(-static_cast<float>(deltaX) * sensitivity), UP);
    direction = glm::vec3(yawRot * glm::vec4(direction, 0.0f));

    //Recompute right after yaw rotation
    right = normalize(cross(direction, UP));

    //Apply pitch (around local right)
    glm::mat4 pitchRot = rotate(glm::mat4(1.0f), glm::radians(-static_cast<float>(deltaY) * sensitivity), right);
    direction = normalize(glm::vec3(pitchRot * glm::vec4(direction, 0.0f)));

    right = normalize(cross(direction, UP));
    up = normalize(cross(right, direction));

    horizontal = right * length(horizontal);
    vertical = up * length(vertical);

    //Scaled by deltaTime
    float movementSpeed = 20.0f * deltaTime;
    if (inputTracker.isKeyHeld(GLFW_KEY_LEFT_SHIFT)) //Sprinting
        movementSpeed *= 10.0f;

    if (inputTracker.isKeyHeld(GLFW_KEY_W))
        position += direction * movementSpeed;

    if (inputTracker.isKeyHeld(GLFW_KEY_S))
        position -= direction * movementSpeed;

    if (inputTracker.isKeyHeld(GLFW_KEY_A))
        position -= normalize(cross(direction, UP)) * movementSpeed;

    if (inputTracker.isKeyHeld(GLFW_KEY_D))
        position += normalize(cross(direction, UP)) * movementSpeed;

    if (inputTracker.isKeyHeld(GLFW_KEY_E))
        position += UP * movementSpeed;

    if (inputTracker.isKeyHeld(GLFW_KEY_Q))
        position -= UP * movementSpeed;

    cameraData.direction = direction;
    cameraData.position = position;
    cameraData.horizontal = horizontal;
    cameraData.vertical = vertical;

    float epsylon = 0.001f;
    cameraData.isMoving = !all(epsilonEqual(oldDirection, direction, epsylon)) || !all(epsilonEqual(oldPosition, position, epsylon));
}

const CameraData& PerspectiveCamera::getCameraData() const {
    return cameraData;
}
