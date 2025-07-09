﻿#include "InputTracker.h"

#include "imgui.h"

InputTracker::InputTracker(GLFWwindow* window) : window(window), deltaX(0.0), deltaY(0.0), lastX(0.0), lastY(0.0) {
    glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (action == GLFW_PRESS) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                ImGuiIO& io = ImGui::GetIO();
                io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
            } else if (action == GLFW_RELEASE) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                ImGuiIO& io = ImGui::GetIO();
                io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
            }
        }
    });

}

void InputTracker::update() {
    // update key states
    for (int key = 0; key <= GLFW_KEY_LAST; ++key) {
        prevKeyStates[key] = keyStates[key];
        keyStates[key] = glfwGetKey(window, key) == GLFW_PRESS;
    }

    //Update mouse button states
    for (int button = GLFW_MOUSE_BUTTON_1; button <= GLFW_MOUSE_BUTTON_LAST; ++button) {
        prevMouseButtonStates[button] = mouseButtonStates[button];
        mouseButtonStates[button] = glfwGetMouseButton(window, button) == GLFW_PRESS;
    }

    //Update mouse position and calculate delta
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    deltaX = xpos - lastX;
    deltaY = ypos - lastY;
    lastX = xpos;
    lastY = ypos;
}

bool InputTracker::isKeyPressed(int key) const {
    return keyStates[key] && !prevKeyStates[key];
}

bool InputTracker::isKeyHeld(int key) const {
    return keyStates[key];
}

bool InputTracker::isKeyReleased(int key) const {
    return !keyStates[key] && prevKeyStates[key];
}

bool InputTracker::isMouseButtonPressed(int button) const {
    return mouseButtonStates[button] && !prevMouseButtonStates[button];
}

bool InputTracker::isMouseButtonHeld(int button) const {
    return mouseButtonStates[button];
}

bool InputTracker::isMouseButtonReleased(int button) const {
    return !mouseButtonStates[button] && prevMouseButtonStates[button];
}

void InputTracker::getMousePosition(double& xpos, double& ypos) const {
    glfwGetCursorPos(window, &xpos, &ypos);
}

void InputTracker::getMouseDelta(double& outDeltaX, double& outDeltaY) const {
    outDeltaX = deltaX;
    outDeltaY = deltaY;
}
