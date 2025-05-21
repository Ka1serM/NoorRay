#pragma once

#include <GLFW/glfw3.h>

class InputTracker {
public:
    InputTracker(GLFWwindow* window);

    void update();

    bool isKeyPressed(int key) const;
    bool isKeyHeld(int key) const;
    bool isKeyReleased(int key) const;

    bool isMouseButtonPressed(int button) const;
    bool isMouseButtonHeld(int button) const;
    bool isMouseButtonReleased(int button) const;

    void getMousePosition(double& xpos, double& ypos) const;

    void getMouseDelta(double& outDeltaX, double& outDeltaY) const;

private:
    GLFWwindow* window;

    bool keyStates[GLFW_KEY_LAST + 1] = { false }; // Initializes all to false
    bool prevKeyStates[GLFW_KEY_LAST + 1] = { false }; // Initializes all to false

    bool mouseButtonStates[GLFW_MOUSE_BUTTON_LAST + 1] = { false }; // Initializes all to false
    bool prevMouseButtonStates[GLFW_MOUSE_BUTTON_LAST + 1] = { false }; // Initializes all to false

    double deltaX, deltaY;
    double lastX, lastY;
};